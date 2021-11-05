import numpy as np
from mnist_local_generator import mnist_localization_generator
from fast_overfeat import overfeat_loss, simple_backbone, cls_header, reg_header, metric_mse, metric_cee
from utils import xywh2xyxy, images_with_rectangles, plot_images, original_rectangle_coords
from tensorflow.keras.models import load_model
from nms import non_maximum_suppression

# Generate Mnist data for localization
_, (test_xs_, _, _) = mnist_localization_generator((100, 100), (100, 100), background=True, n_sample=1000)

# model laod
model = load_model('../models/model.h5',
                   custom_objects={"overfeat_loss": overfeat_loss, "metric_mse": metric_mse, "metric_cee": metric_cee})
# prediction
pred = model.predict(test_xs_ / 255.)
n_classes = 11
pred_cls = pred[..., :n_classes]
pred_reg = pred[..., n_classes:]
pred_reg = xywh2xyxy(pred_reg)

# feature map cell 에 대한  original 위치 좌표를 복원
fmap_size = (2, 2)
kernel_sizes = [5, 2, 5, 2, 3, 3, 3, 2, 4, 1, 1]
strides = [2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1]
paddings = ['valid', 'valid', 'valid', 'valid', 'same', 'same', 'same', 'valid', 'valid', 'valid', 'valid']
coords_xywh = original_rectangle_coords(fmap_size, kernel_sizes, strides, paddings)
coords_xyxy = xywh2xyxy(coords_xywh)

grids = [coords_xyxy] * 100
images_with_grids = images_with_rectangles(test_xs_[..., 0], grids)

# regression offset 계산, offset shape : (feature_map_size, 4)
x1y1_coords = coords_xyxy[:, ::-1][:, -2:]
offset = np.concatenate([x1y1_coords, x1y1_coords], axis=-1)

# reshape
pred_reg = pred_reg.reshape(-1, 4, 4)
pred_cls = pred_cls.reshape(-1, 4, n_classes)

# offset 적용
pred_reg_shift = pred_reg + offset

# visualization
bboxed_imgs_shift = images_with_rectangles(test_xs_[..., 0], pred_reg_shift)
bboxed_imgs_shift = np.array(bboxed_imgs_shift)

# 모든 이미지에 NMS을 적용
pred_reg_bucket = []
for i in range(len(pred_reg_shift)):
    pred_reg_nms, pred_cls_nms = non_maximum_suppression(pred_reg_shift[i], pred_cls[i], 0.5)
    pred_reg_bucket.append(pred_reg_nms)

rected_test_xs = images_with_rectangles(test_xs_, pred_reg_bucket)
plot_images(rected_test_xs[:100])
