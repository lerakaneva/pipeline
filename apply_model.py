import tensorflow as tf
import numpy as np
import tifffile as tiff
import cv2
import os
import gc
import glob
from tqdm import tqdm

# Constants for parameters
RESIZE_DIMENSIONS = (256, 256)
FINAL_OUTPUT_DIMENSIONS = (512, 512)
BATCH_SIZE = 32

def load_and_preprocess_tiff(filename):
    """Loads and preprocesses the TIFF stack for segmentation.

    Args:
        filename: The path to the input TIFF file.
    
    Returns:
        np.ndarray: Preprocessed TIFF stack ready for segmentation.
    """
    raw_data = tiff.imread(filename)
    processed_stack = [
        preprocess_slide(slide) for slide in raw_data
    ]
    return np.expand_dims(np.array(processed_stack, dtype=np.float32), axis=-1)

def preprocess_slide(slide):
    """Resizes and normalizes a single slide from the TIFF stack.

    Args:
        slide: A single TIFF slide to be processed.
    
    Returns:
        np.ndarray: Preprocessed slide.
    """
    resized_slide = cv2.resize(slide, RESIZE_DIMENSIONS, interpolation=cv2.INTER_AREA)
    normalized_slide = (resized_slide - resized_slide.min()) / (resized_slide.max() - resized_slide.min())
    return normalized_slide

def run_inference(model, processed_stack):
    """Runs inference on the preprocessed TIFF stack using the segmentation model.

    Args:
        model: The TensorFlow model for segmentation.
        processed_stack: The preprocessed TIFF stack.
    
    Returns:
        np.ndarray: Predicted mask stack from the model.
    """
    infer = model.signatures['serving_default']
    input_name = list(infer.structured_input_signature[1].keys())[0]
    num_batches = (processed_stack.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    # Assuming the model's output shape is the same as the input shape
    # Adjust output shape
    output_shape = (processed_stack.shape[0], processed_stack.shape[1], processed_stack.shape[2], 3)
    predicted_masks = np.empty(output_shape, dtype=np.float32)


    with tqdm(total=num_batches, desc="Processing Batches", unit="batch") as pbar:
        for i in range(num_batches):
            batch = processed_stack[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            batch_result = infer(**{input_name: tf.constant(batch)})
            output_key = list(batch_result.keys())[0]
            predicted_masks[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = batch_result[output_key].numpy()
            pbar.update(1)
    
    gc.collect()
    return predicted_masks


def predict_for_file(filename, model):
    """Loads and preprocesses tiff, runs inference using the segmentation model.
    Args:
        filename: The path to the input TIFF file.
        model: The TensorFlow model for segmentation.
    Returns:
        np.ndarray: Predicted mask stack from the model.
    """
    # Load and preprocess TIFF stack
    processed_stack = load_and_preprocess_tiff(filename)
    # Run model inference
    predicted_masks = run_inference(model, processed_stack)
    return predicted_masks

def remap_mask_values(predicted_masks):
    """Remaps the predicted mask values to the desired output format, inplace.

    Args:
        predicted_masks: Predicted mask stack.

    Returns:
        None (modifies the input array inplace)
    """
    predicted_masks[predicted_masks == 1] = 3
    predicted_masks[predicted_masks == 2] = 1
    predicted_masks[predicted_masks == 0] = 2
    predicted_masks[predicted_masks == 3] = 0 


def save_segmented_stack(output_path, input_file_path, segmented_stack):
    """Saves the segmented mask stack as a TIFF file.

    Args:
        output_path: The directory to save the segmented TIFF file.
        filename: The original input filename.
        segmented_stack: The segmented mask stack.
    """
    output_tiff_path = os.path.join(output_path, os.path.basename(input_file_path))
    tiff.imwrite(output_tiff_path, segmented_stack)
    print(f"Segmented stack saved to {output_tiff_path}")

def segment_file(model, filename, output_path):
    """Segments a single TIFF file using the provided model.

    Args:
        model: The TensorFlow model for segmentation.
        filename: The path to the input TIFF file.
        output_path: The path to the output directory for segmented TIFF files.
    """
    print(f"Processing {filename}")

    predicted_masks = predict_for_file(filename, model)
    
    gc.collect()

    # Convert class probabilities to class
    predicted_masks = np.argmax(predicted_masks, axis=-1)
        
    # Remap mask values and resize for output
    remap_mask_values(predicted_masks)
    predicted_masks = np.array([cv2.resize(mask, FINAL_OUTPUT_DIMENSIONS, interpolation=cv2.INTER_NEAREST).astype(np.uint8) for mask in predicted_masks])

    # Save the segmented stack
    save_segmented_stack(output_path, filename, predicted_masks)

    gc.collect()

def apply_model_tiff_files(model, input_path, output_path):
    """Apply model to all TIFF files in the input directory.

    Args:
        model: The TensorFlow model for segmentation.
        input_path: The path to the directory containing input TIFF files.
        output_path: The path to the output directory for segmented TIFF files.
    """
    tif_files = glob.glob(os.path.join(input_path, '**', '*.tif'), recursive=True)
    os.makedirs(output_path, exist_ok=True)
    for tif_file in tif_files:
        segment_file(model=model, filename=tif_file, output_path=output_path)

if __name__ == "__main__":
    # Define the input, output, and model directories
    input_folder =  'path/to/input_folder'
    output_folder = 'path/to/output_folder'
    path_to_model = 'path/to/model'

    # Load the model
    model = tf.saved_model.load(path_to_model)

    # Process the TIFF files
    apply_model_tiff_files(model, input_folder, output_folder)
