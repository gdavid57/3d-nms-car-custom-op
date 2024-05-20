# 3D Non Max Suppression and Crop And Resize Custom Operation for TensorFlow 2.3

This GitHub repository contains the code to compile the 3D Non Max Suppression and Crop And Resize custom operations for TensorFlow 2.3 GPU. It is based on the procedure for custom op of TensorFlow (see here for detailed informations: https://github.com/tensorflow/custom-op.git).

## Installation

To install the custom operations, follow the instructions below:

1. Clone this repository by running the following command:

```
git clone https://github.com/gdavid57/3d-nms-car-custom-op.git
```

2. Navigate into the cloned repository:
```
cd 3d-nms-car-custom-op/
```

3. Make the generate_whl.sh script executable:

```
chmod +rwx generate_whl.sh
```

4. Run the Docker container with GPU support and mount the current working directory:

```
docker run --gpus all -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16 bash /working_dir/generate_whl.sh
```
After the compilation process completes, the output package will be available in the artefact/ directory.

5. Install the generated .whl package using pip in an environment with TensorFlow 2.3 GPU:

```
pip install artefact/<package-name>.whl
```

## Test operations

We provide tests for each operation included in this repository. These tests are directly inspired by the tests found in TensorFlow sources for their two-dimensional counterparts. We compare our 3D implemementation of the Crop And Resize op with a method based on the scipy.interpolate.RegularGridInterpolator function.

To run the tests in your installation environment, just do:

```
python crop_and_resize_3d/python/ops/crop_and_resize_3d_ops_test.py
python non_max_suppression_3d/python/ops/non_max_suppression_3d_ops_test.py
```

Note: two tests of the Crop And Resize appear as "not Ok" but actually are. The difference of results between our 3D Crop And Resize and the scipy.interpolate.RegularGridInterpolator simply highlights that the choices made by these two methods of "what is nearest?" is not the same in this very particular case.

## Additional Information

These custom operations are part of our project 3D Mask R-CNN. For more information about it, please refer to the following repository:

3D Mask R-CNN Repository: https://github.com/gdavid57/3d-mask-r-cnn

For any issues or questions related to the custom operations, please open an issue in this repository.