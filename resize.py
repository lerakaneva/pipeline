import os
import glob
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image, ImageSequence
from tqdm import tqdm

# Configuration constants
IMAGE_SIZE = (512, 512)

def convert_rgba_to_rgb(image):
    """
    Converts an RGBA image to RGB if needed.
    
    Args:
        image: Input image.

    Returns:
        np.ndarray: Converted RGB image.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def read_tif_frames(tif_file_path):
    """
    Reads a multi-frame .tif file and converts each frame into a NumPy array.
    
    Args:
        tif_file_path (str): The path to the .tif file to be read.

    Returns:
        list of np.ndarray: A list of frames where each frame is a NumPy array.
                            The frames are converted to RGB if they were in RGBA format.
                            Returns None if an error occurs.
    """
    try:
        with Image.open(tif_file_path) as img:
            frames = [
                convert_rgba_to_rgb(np.array(frame)) 
                for frame in ImageSequence.Iterator(img)
            ]
            return frames
    except Exception as e:
        print(f"Failed to read {tif_file_path}: {e}")
        return None
    
def normalize_image_intensity(image):
    """
    Normalizes the pixel values of an image to the range [0, 1].
    
    Args:
        image: The input image as a NumPy array.

    Returns:
        np.ndarray: The normalized image where pixel values are in the range [0, 1].
    """
    image = image.astype(np.float32)
    min_val = image.min()
    image -= min_val
    max_val = image.max()
    
    if max_val != 0:
        image /= max_val
        
    return image

def resize_image(image, target_size):
    """
    Resizes an image to a specified size.
    
    Args:
        image: The image to be resized.
        target_size (tuple): The target size as (width, height).

    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_tif_file(tif_file_path):
    """
    Processes a .tif file by reading frames, resizing, and normalizing each frame.
    
    Args:
        tif_file_path: The path to the .tif file to be processed.

    Returns:
        list of np.ndarray: A list of resized and normalized frames.
    """
    frames = read_tif_frames(tif_file_path)
    if frames is None:
        return []
    
    processed_frames = []
    for frame in frames:
        if frame.shape[:2] != IMAGE_SIZE:
            frame = resize_image(frame, IMAGE_SIZE)
        normalized_frame = normalize_image_intensity(frame)
        processed_frames.append(normalized_frame)
    
    return processed_frames

def validate_frame_shapes(frames):
    """
    Validates that all frames in the list have the same shape.
    
    Args:
        frames (list of np.ndarray): List of frames to validate.
    
    Returns:
        bool: True if all frames have the same shape, False otherwise.
    """
    if not frames:
        return False
    
    first_shape = frames[0].shape
    for frame in frames:
        if frame.shape != first_shape:
            print(f"Frame with mismatched shape found: {frame.shape} != {first_shape}")
            return False
    return True

def save_tif_stack(frames, output_file_path):
    """
    Saves a stack of frames into a single .tif file, handling errors and checking for consistency.
    
    Args:
        frames (list of np.ndarray): A list of processed frames to be saved.
        output_file_path: The path to save the output .tif file.
    """
    if not validate_frame_shapes(frames):
        print(f"Skipping {output_file_path}: Inconsistent frame sizes")
        return
    
    try:        
        with tiff.TiffWriter(output_file_path, ome=False, append=False) as tif_writer:
            for frame in tqdm(frames, desc="Saving frames", total=len(frames)):
                # Convert frame to uint8 as required
                frame_uint8 = (frame * 255).astype(np.uint8)
                tif_writer.write(frame_uint8, contiguous=True)
        
        print(f"Saved {len(frames)} frames to {output_file_path}")
    
    except Exception as e:
        print(f"Error while saving to {output_file_path}: {e}")


def process_all_tif_files(input_folder, output_folder):
    """
    Processes all .tif files in the input folder and saves the results to the output folder.
    
    Args:
        input_folder: Path to the folder containing .tif files.
        output_folder: Path to the folder where processed .tif files will be saved.
    """
    tif_files = glob.glob(os.path.join(input_folder, '**', '*.tif'), recursive=True)
    
    if not tif_files:
        print("No .tif files found in the input folder.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    for tif_file in tqdm(tif_files, desc="Processing .tif files"):
        output_file_path = os.path.join(output_folder, os.path.basename(tif_file))
        processed_frames = process_tif_file(tif_file)
        if processed_frames:
            save_tif_stack(processed_frames, output_file_path)

if __name__ == "__main__":
    input_folder = 'path/to/input_folder'
    output_folder = 'path/to/output_folder'
    
    process_all_tif_files(input_folder, output_folder)


