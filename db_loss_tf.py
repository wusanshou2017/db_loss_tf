# -*- coding: utf-8 -*-
# @Time : 2022\8\17  13:26
# @Author : tdwu
# @Email :wu_allin@sina.com


import tensorflow as tf
import numpy as np


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    # weighted element-wise losses
    if weight is not None:
        weight = tf.cast(weight, tf.float32)
    else:
        weight = 1.0

    loss = tf.losses.sigmoid_cross_entropy(
        label, pred, weights=weight, reduction='none')

    # loss = tf.keras.losses.binary_crossentropy(label, pred, from_logits=True,label_smoothing=weight)
    if reduction == "none":
        return loss
    loss = tf.reduce_mean(loss)

    return loss


class ResampleLoss():

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None,  # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(CB_beta=0.9, CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                              ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.cls_criterion = binary_cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha']  # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = tf.convert_to_tensor((np.asarray(class_freq)), dtype=tf.float32)
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num  # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - tf.math.log(
            self.train_num / self.class_freq - 1) * init_bias  ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = tf.ones(self.class_freq.shape) / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label, cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score, label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            # print("logpt:...", logpt)
            pt = tf.exp(-logpt)

            # print("weight:...", weight)
            wtloss = self.cls_criterion(
                cls_score, tf.cast(label, tf.float32), weight=weight, reduction='none')
            # print("wtloss:...", wtloss)
            alpha = tf.fill(tf.shape(label),self.alpha)
            diff_alpht = tf.fill(tf.shape(label),1-self.alpha)
            alpha_t = tf.where(tf.equal(label,1.0),alpha, diff_alpht)
            # print("alpha_t:...", alpha_t)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss  ####################### balance_param should be a tensor
            # print("pt:...", pt)
            # print("loss:...", loss)
            loss = tf.reduce_mean(loss)  ############################ add reduction
            # print("loss:...", loss)
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(tf.cast(label, tf.float32))
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(tf.cast(label, tf.float32))
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(tf.cast(label, tf.float32))
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = tf.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / tf.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels * 1.0) * self.neg_scale + logits * labels * 1.0
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels * 1.0) + weight * labels * 1.0
        return logits, weight

    def rebalance_weight(self, gt_labels):
        print("gt_labels:...", gt_labels)
        print("self_freq_inv:...", self.freq_inv)
        repeat_rate = tf.reduce_sum(tf.cast(gt_labels, tf.float32) * self.freq_inv, axis=1, keepdims=True)
        # pos_weight = self.freq_inv.unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        pos_weight = tf.expand_dims(self.freq_inv, axis=0) / repeat_rate
        weight = tf.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 引入placeholder,取消eager模式
    # tf.enable_eager_execution()
    # print(tf.executing_eagerly())

    # if loss_func_name == 'DBloss': # DB
    loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                             class_freq=[0.3, 0.2, 0.5], train_num=2)
    #
    logits = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]],dtype=tf.float32)
    labels = tf.constant([[1, 0, 1], [1, 0, 1]],dtype=tf.float32)
    cls_score = tf.placeholder(tf.float32,[2,3])
    label = tf.placeholder(tf.float32,[2,3])
    logits = [[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]
    labels = [[1, 0, 1], [1, 0, 1]]
    # num_labels = 3
    # loss = loss_func.forward(logits, tf.cast(labels, tf.float32))
    #
    # print("loss:", loss)
    loss =loss_func.forward(cls_score,label)
    with tf.Session() as sess:
        l=sess.run(loss,feed_dict={cls_score:logits,label:labels})
        print(l)
