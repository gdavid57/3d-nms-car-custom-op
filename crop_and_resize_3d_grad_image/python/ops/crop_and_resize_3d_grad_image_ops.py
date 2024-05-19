from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


crop_and_resize_3d_grad_image_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_crop_and_resize_3d_grad_image_op.so'))

crop_and_resize_3d_grad_image = crop_and_resize_3d_grad_image_ops.crop_and_resize3d_grad_image

