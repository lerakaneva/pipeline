import os
from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import tifffile as tiff
from joblib import Parallel, delayed

# Threshold constants
SMALL_AREA_THRESHOLD = 40
LARGE_AREA_THRESHOLD = 200
HIGH_CIRCULARITY_THRESHOLD = 0.6
LOW_CIRCULARITY_THRESHOLD = 0.2


def load_image_stack(path):
    """
    Load an image stack from the specified path.

    Args:
        path (str): The path to the image stack.

    Returns:
        np.ndarray: The loaded image stack.

    Raises:
        IOError: If the image stack fails to load.
    """
    try:
        return tiff.imread(path)
    except IOError as e:
        print(f"Failed to load image stack from {path}: {e}")
        raise


def calculate_circularity(area, perimeter):
    """
    Calculate the circularity of an object.

    Circularity is defined as (4 * π * area) / (perimeter^2).

    Args:
        area (float): The area of the object.
        perimeter (float): The perimeter of the object.

    Returns:
        float: The circularity of the object. Returns 0 if perimeter is 0.
    """
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0


def extract_features(image_array, frame_idx):
    """
    Extract features such as area, perimeter, and circularity from a single frame of the image stack.

    Args:
        image_array (np.ndarray): A 2D array representing a single frame of the image stack.
        frame_idx (int): The index of the current frame.

    Returns:
        tuple: A tuple containing:
            - List[Dict]: A list of dictionaries, where each dictionary contains features of an object.
            - Dict: A dictionary of region properties.
    """
    labeled_image = label(image_array > 0, connectivity=2)
    props_table = regionprops_table(labeled_image, intensity_image=image_array,
                                    properties=('label', 'centroid', 'area', 'perimeter'))
    props = regionprops(labeled_image)
    features = []

    for i in range(len(props_table['label'])):
        circularity = calculate_circularity(props_table['area'][i], props_table['perimeter'][i])
        features.append({
            'frame': frame_idx,
            'y': props_table['centroid-0'][i],
            'x': props_table['centroid-1'][i],
            'area': props_table['area'][i],
            'perimeter': props_table['perimeter'][i],
            'id': props_table['label'][i],
            'circularity': circularity,
            'coords': props[i].coords
        })

    return features, props_table


def calculate_dominant_color(image, coords):
    """
    Calculate the dominant color of an object based on its pixels.

    Args:
        image (np.ndarray): The original image.
        coords (np.ndarray): The coordinates of the object in the image.

    Returns:
        int: The dominant color of the object based on the most frequent pixel value.
    """
    object_pixels = image[coords[:, 0], coords[:, 1]]
    values, counts = np.unique(object_pixels, return_counts=True)
    dominant_color = values[np.argmax(counts)]
    return dominant_color


def color_objects_by_features(image_array, properties):
    """
    Apply colors to objects in the image based on their features (area, circularity).

    Args:
        image_array (np.ndarray): The original 2D image.
        properties (list[dict]): A list of features for each detected object.

    Returns:
        np.ndarray: The modified image with colored objects.
    """
    colored_image = np.copy(image_array)
    for feature in properties:
        circularity = feature['circularity']
        area = feature['area']
        coords = feature['coords']

        if area < SMALL_AREA_THRESHOLD:
            fill_color = 0
        elif circularity > HIGH_CIRCULARITY_THRESHOLD and area < LARGE_AREA_THRESHOLD:
            fill_color = 2
        elif circularity < LOW_CIRCULARITY_THRESHOLD or area > LARGE_AREA_THRESHOLD:
            fill_color = 1
        else:
            fill_color = calculate_dominant_color(image_array, coords)
        
        colored_image[coords[:, 0], coords[:, 1]] = fill_color

    return colored_image


def process_single_frame(image, frame_idx):
    """
    Process a single frame of the image stack by extracting features and applying coloring.

    Args:
        image (np.ndarray): The image frame to be processed.
        frame_idx (int): The index of the current frame.

    Returns:
        np.ndarray: The processed frame with applied colors based on object features.
    """
    features, _ = extract_features(image, frame_idx)
    return color_objects_by_features(image, features)


def save_image_stack(processed_stack, input_file_path, output_path):
    """
    Save the processed image stack to the specified output directory.

    Args:
        processed_stack (np.ndarray): The processed image stack to be saved.
        input_file_path (str): The path of the original input file.
        output_path (str): The directory where the output file will be saved.
    """
    output_tiff_path = os.path.join(output_path, os.path.basename(input_file_path))
    tiff.imwrite(output_tiff_path, processed_stack)
    print(f'Saved the processed stack to {output_tiff_path}')


def correct_segmentation_file(input_file_path, output_path):
    """
    Process a segmentation file by extracting features, applying color, and saving the result.

    Args:
        input_file_path (str): The path to the input segmentation file.
        output_path (str): The directory where the corrected file will be saved.
    """
    image_stack = load_image_stack(input_file_path)
    processed_stack = [process_single_frame(image, idx) for idx, image in enumerate(image_stack)]
    
    save_image_stack(np.array(processed_stack), input_file_path, output_path)


def correct_segmentation_files_in_a_folder(input_path, output_path, n_jobs=-1):
    """
    Process all segmentation files in a folder and save the results.

    Args:
        input_path (str): The path to the folder containing segmentation files.
        output_path (str): The directory where the corrected files will be saved.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1 (use all processors).
    """
    tif_files = [str(tif_file) for tif_file in Path(input_path).rglob('*.tif')]
    os.makedirs(output_path, exist_ok=True)
    Parallel(n_jobs=n_jobs)(delayed(correct_segmentation_file)(tif_file, output_path) 
                            for tif_file in tif_files)


def main():
    """
    Main function to process segmentation files in the specified input folder and save them to the output folder.
    """

    input_folder = 'path/to/input_folder'
    output_folder = 'path/to/output_folder'

    correct_segmentation_files_in_a_folder(input_path=input_folder, output_path=output_folder)


if __name__ == "__main__":
    main()
