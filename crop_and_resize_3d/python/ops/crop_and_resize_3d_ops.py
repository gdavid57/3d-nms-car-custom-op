from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


crop_and_resize_3d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_crop_and_resize_3d_ops.so'))

crop_and_resize_3d = crop_and_resize_3d_ops.crop_and_resize3d