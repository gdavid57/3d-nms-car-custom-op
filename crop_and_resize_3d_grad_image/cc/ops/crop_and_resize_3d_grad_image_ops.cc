#define EIGEN_USE_THREADS

#include <functional>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("CropAndResize3DGradImage")
    .Input("grads: float")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("image_size: int32")
    .Output("output: T")
    .Attr("T: {float, half, double}")
    .Attr("method_name: {'trilinear', 'nearest'} = 'trilinear'")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      TF_RETURN_IF_ERROR(c->WithRank(out, 5, &out));
      c->set_output(0, out);
      return Status::OK();
    });