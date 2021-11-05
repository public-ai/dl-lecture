from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Softmax
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, MSE


def overfeat_loss(true, pred):
    """

    :param true: ndarray, 4d tensor (NHWC) 단 C = n_classes + 4
        regresssion + classification 순서로 concatenate 됨
    :param pred: ndarray, 4d tensor (NHWC) 단 C= n_classes + 4
        regresssion + classification 순서로 concatenate 됨
    :return: total_error, float
    :return:
    """
    n_reg = 4

    # slicing classification and regression
    pred_reg = pred[..., :n_reg]
    pred_cls = pred[..., n_reg:]
    true_reg = true[..., :n_reg]
    true_cls = true[..., n_reg:]

    # Generate background class negative, foreground class positive
    mask = true_cls[..., 10] != 1

    masked_pred_reg = pred_reg[mask]
    masked_true_reg = true_reg[mask]

    cee_ = overfeat_cee(true_cls, pred_cls)
    mse_ = overfeat_mse(masked_true_reg, masked_pred_reg)
    total_error = cee_ + mse_

    # background 정보는 regression은 학습시키지 않는다.

    return total_error


def overfeat_mse(true, pred):
    """

    :param true: ndarray, 4d tensor (NHWC) 단 C=4
    :param pred: ndarray or tensor, 4d tensor (NHWC), 단 C=4
    :return: cee_, float,
    """

    # slicing classification and regression
    mse = MSE(true, pred)
    mse_ = tf.math.reduce_mean(mse)

    return mse_


def overfeat_cee(true, pred):
    """
    :param true: ndarray, 4d tensor (NHWC), C= number of classes
    :param pred: ndarray or tensor, 4d tensor (NHWC), C= number of classes
    :return: cee_, float,
    """

    cee = CategoricalCrossentropy()
    cee_ = cee(true, pred)

    return cee_


def metric_mse(true, pred):
    """

    :param true: ndarray, 4d tensor (NHWC) 단 C=4
    :param pred: ndarray or tensor, 4d tensor (NHWC), 단 C=4
    :return: cee_a, float,
    """
    n_reg = 4

    # slicing classification and regression
    pred_reg = pred[..., :n_reg]
    true_reg = true[..., :n_reg]

    # slicing classification and regression
    mse = MSE(true_reg, pred_reg)
    mse_ = tf.math.reduce_mean(mse)

    return mse_


def metric_cee(true, pred):
    """
    :param true: ndarray, 4d tensor (NHWC), C= number of classes
    :param pred: ndarray or tensor, 4d tensor (NHWC), C= number of classes
    :return: cee_, float,
    """
    n_reg = 4

    # slicing classification and regression
    pred_cls = pred[..., n_reg:]
    true_cls = true[..., n_reg:]

    cee = CategoricalCrossentropy()
    cee_ = cee(true_cls, pred_cls)

    return cee_


def simple_backbone(input_):
    # Keras Input

    # conv block 1
    layer = Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu',
                   kernel_initializer='he_normal')(input_)
    layer = MaxPool2D(strides=2)(layer)

    # conv block 2
    layer = Conv2D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu',
                   kernel_initializer='he_normal')(layer)
    layer = MaxPool2D(strides=2)(layer)

    # conv block 3
    layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')(layer)

    # conv block 4
    layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')(layer)

    # conv block 5
    layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal')(layer)
    layer = MaxPool2D(strides=2)(layer)
    return layer


def cls_header(backbone_layer, add_background=True):
    fc1_cls = Conv2D(kernel_size=4, filters=512, activation='relu', kernel_initializer='he_normal')(backbone_layer)
    fc2_cls = Conv2D(kernel_size=1, filters=512, activation='relu', kernel_initializer='he_normal')(fc1_cls)
    if add_background:
        num_classes = 11
    else:
        num_classes = 10
    cls_pred = Conv2D(kernel_size=1, filters=num_classes, activation='softmax')(fc2_cls)
    return cls_pred


def reg_header(backbone_layer):
    # fully connected layer / Regression
    fc1_reg = Conv2D(kernel_size=4, filters=512, activation='relu', kernel_initializer='he_normal')(backbone_layer)
    fc2_reg = Conv2D(kernel_size=1, filters=512, activation='relu', kernel_initializer='he_normal')(fc1_reg)
    reg_pred = Conv2D(kernel_size=1, filters=4, activation=None)(fc2_reg)
    return reg_pred
