from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, Concatenate
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from mnist_local_generator import mnist_localization_generator
from fast_overfeat import overfeat_loss, simple_backbone, cls_header, reg_header, metric_mse, metric_cee
from utils import xywh2xyxy, images_with_rectangles, plot_images

# Generate Mnist data for localization

(train_xs, train_cls_ys, train_reg_ys), (test_xs, test_cls_ys, test_reg_ys) = \
    mnist_localization_generator((84, 84), (84, 84), background=True, n_sample=100)
train_ys = np.concatenate([train_cls_ys, train_reg_ys], axis=-1)
test_ys = np.concatenate([test_cls_ys, test_reg_ys], axis=-1)

# Model
K.clear_session()
input_ = Input(shape=(None, None, 1))
backbone_layer = simple_backbone(input_)
pred_cls = cls_header(backbone_layer)
pred_reg = reg_header(backbone_layer)
pred = tf.concat([pred_cls, pred_reg], axis=-1)

# model
model = Model(input_, pred)
model.compile('adam', loss=overfeat_loss, metrics=['acc'])
model.fit(x=train_xs / 255., y=train_ys, epochs=5, validation_data=(test_xs / 255., test_ys))

# mode
pred = model.predict(test_xs / 255.)
pred_ori = pred.copy()

n_classes = 11
pred_cls = pred[..., :n_classes]
pred_reg = pred[..., n_classes:]

pred_reg = pred_reg.reshape(-1, 1, 4)
pred_reg = xywh2xyxy(pred_reg)

bboxed_imgs = images_with_rectangles(test_xs[..., 0], pred_reg)
bboxed_imgs = np.array(bboxed_imgs)

plot_images(bboxed_imgs[:100])
