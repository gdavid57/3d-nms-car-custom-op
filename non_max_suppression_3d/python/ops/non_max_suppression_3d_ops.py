from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


non_max_suppression_3d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_non_max_suppression_3d_ops.so'))

non_max_suppression_3d = non_max_suppression_3d_ops.non_max_suppression3d
