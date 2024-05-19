#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("NonMaxSuppression3D")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Output("selected_indices: int32")
    .Attr("iou_threshold: float = 0.5")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Get inputs and validate ranks.
      ::tensorflow::shape_inference::ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
      ::tensorflow::shape_inference::ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
      ::tensorflow::shape_inference::ShapeHandle max_output_size;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      ::tensorflow::shape_inference::DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 6.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 6, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });