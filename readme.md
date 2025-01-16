# Platelet Segmentation Pipeline

This pipeline segments TIFF image stacks to identify thrombi and platelets. It consists of three steps: resizing and normalization, neural network segmentation, and segmentation correction.

## Prerequisites (Apple Processors like M3)

* **Python 3.9:** Ensure Python 3.9 is installed.
* **HDF5:** Install the HDF5 library:
```bash
  brew install hdf5
```

* **Required Libraries:** Install the necessary Python packages: 
```python
pip install -r requirements.txt
```
## Running the Pipeline

The pipeline consists of three Python scripts that should be executed in the following order:

1. **Resize and Normalize (`resize.py`):**
   - Modify `input_folder` and `output_folder` within `resize.py`.
   - `input_folder`: Path to the directory containing the original TIFF stacks.
   - `output_folder`: Path to the directory where resized and normalized TIFF stacks will be saved.
   - Run

2. **Apply Segmentation Model (`apply_model.py`):**
   - Modify `input_folder`, `output_folder`, and `path_to_model` within `apply_model.py`.
   - `input_folder`: Path to the directory containing the resized TIFF stacks (output of `resize.py`).
   - `output_folder`: Path to the directory where the segmented TIFF stacks (masks) will be saved.
   - `path_to_model`: Path to the TensorFlow saved model. Ensure your model is saved using:
     ```python
     tf.saved_model.save(model, new_path_to_model)
     ```
   - Run

3. **Correct Segmentation (`correct.py`):**
   - Modify `input_folder` and `output_folder` within `correct.py`.
   - `input_folder`: Path to the directory containing the segmented TIFF stacks (output of `apply_model.py`).
   - `output_folder`: Path to the directory where the corrected TIFF stacks will be saved.
   - Run


## Script Descriptions

1. **`resize.py`:** Resizes and normalizes TIFF image stacks.  It reads multi-frame TIFF files, resizes each frame to 512x512 pixels, and normalizes pixel intensities to the range [0, 1].  The script handles potential RGBA to RGB conversion and checks for frame size consistency before saving the processed frames back into a TIFF stack.

2. **`apply_model.py`:** Applies a pre-trained TensorFlow saved model to TIFF image stacks for segmentation. It loads the saved model, preprocesses the input TIFF stacks, performs inference in batches for efficiency, and saves the resulting segmentation masks as multi-frame TIFF files.

3. **`correct.py`:** Refines TIFF stacks of segmented images. It analyzes each frame, calculating properties such as area, perimeter, and circularity of labeled regions. Based on these calculated properties, it re-colors regions to correct segmentation errors and distinguish between different objects. The refined stack is then saved as a multi-frame TIFF. Parallel processing is used for improved performance.


