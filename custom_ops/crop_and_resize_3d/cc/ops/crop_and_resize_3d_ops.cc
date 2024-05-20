#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

namespace {

// Sets output[0] to shape [batch_dim,height,width,depth,channel_dim], where
// height and width and depth come from the size_tensor.
Status SetOutputToSizedImage(::tensorflow::shape_inference::InferenceContext* c,
                              ::tensorflow::shape_inference::DimensionHandle batch_dim,
                             int size_input_idx,
                             ::tensorflow::shape_inference::DimensionHandle channel_dim) {
  // Verify shape of size input.
  ::tensorflow::shape_inference::ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  ::tensorflow::shape_inference::DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 3, &unused));

  // Get size values from the size tensor.
  const Tensor* size_tensor = c->input_tensor(size_input_idx);
  ::tensorflow::shape_inference::DimensionHandle width;
  ::tensorflow::shape_inference::DimensionHandle height;
  ::tensorflow::shape_inference::DimensionHandle depth;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
    depth = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
          " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    height = c->MakeDim(vec(0));
    width = c->MakeDim(vec(1));
    depth = c->MakeDim(vec(2));
  }
  c->set_output(0, c->MakeShape({batch_dim, height, width, depth, channel_dim}));
  return Status::OK();
}

}

REGISTER_OP("CropAndResize3D")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_index: int32")
    .Input("crop_size: int32")
    .Output("crops: float")
    .Attr("T: {uint8, uint16, int8, int16, int32, int64, half, float, double}")
    .Attr("method_name: {'trilinear', 'nearest'} = 'trilinear'")
    .Attr("extrapolation_value: float = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Get inputs and validate ranks.
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));
      ::tensorflow::shape_inference::ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));
      ::tensorflow::shape_inference::ShapeHandle box_ind;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &box_ind));

      // boxes[0] and box_ind[0] are both num_boxes.
      ::tensorflow::shape_inference::DimensionHandle num_boxes_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(box_ind, 0), &num_boxes_dim));

      // boxes.dim(1) is 6.
      ::tensorflow::shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 6, &unused));

      return SetOutputToSizedImage(c, num_boxes_dim, 3 /* size_input_idx */,
                                   c->Dim(input, 4));
    });