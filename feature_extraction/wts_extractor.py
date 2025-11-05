import os

from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from loguru import logger
import cv2 as cv
import yaml
import numpy as np
import torch
import clip
import fire

from bbox import get_square_box


def load_configs(config_file_path):
    """Load configurations from the config file.
    Args:
        config_file_path: path to the config file.
    Returns:
        configs: configurations loaded from the file
    """
    with open(config_file_path, 'r', encoding='UTF-8') as file:
        configs = yaml.safe_load(file)
    return configs


def setup_model(clip_model_dir):
    """Setup CLIP model."""
    model, preprocess = clip.load("ViT-L/14", download_root=clip_model_dir)
    model.eval()
    model.to('cpu')  # Move the model to CPU instead of GPU
    return model, preprocess


def ensure_directory_exists(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def process_frame(frame, box_coordinates, preprocess, model, output_path=None):
    """Process each frame and extract features."""
    x1, y1, x2, y2 = box_coordinates
    frame = frame[y1:y2, x1:x2]  # Crop the frame based on box coordinates
    frame = cv.cvtColor(cv.resize(frame, (224, 224)), cv.COLOR_BGR2RGB)  # Resize and convert to RGB
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).to('cpu')  # Ensure the tensor is on the CPU

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()

    if output_path is not None:
        # Save features if output_path is provided
        image_features = np.expand_dims(image_features, axis=0)  # Add batch dimension
        np.save(output_path, image_features)
    else:
        return image_features  # Return the features if output_path is None

    return None


def extract_features(configs, is_external):
    """Extract features for the given configuration.
    Args:
        configs: configurations.
        is_external: whether the dataset is external or internal.
    Returns:
        None
    """
    video_path = configs['paths']['input_path']
    anno_path = configs['paths']['anno_path']
    output_path = configs['paths']['output_path']
    feature_type = configs['feature_type']

    logger.info(f"Video path: {video_path}")
    logger.info(f"Annotation path: {anno_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Feature type: {feature_type}")

    model, preprocess = setup_model(configs['paths']['CLIP_PATH'])
    ensure_directory_exists(output_path)

    # Ensure input_path is a file and process it
    if os.path.isfile(video_path):
      file_list = [video_path]  # Just add the single video file to the list
    else:
      raise ValueError(f"Error: {video_path} is not a valid file!")

    # check video_path exist
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    # check anno_path exist
    if anno_path:
      assert os.path.exists(anno_path), f"Annotation path {anno_path} does not exist."
 
    # check feature_type is valid
    assert feature_type in ['global', 'sub_global', 'local'], f"Invalid feature type {feature_type}."

    with logging_redirect_tqdm():
        if is_external:
            for filename in tqdm(file_list, desc='Processing videos in external dataset'):
                logger.info(f"Processing video: {filename}")
                extract_features_external(filename, video_path, anno_path, output_path, feature_type, preprocess, model)
        else:
            # print("Internal dataset")  # Debug statement
            for view_name in tqdm(file_list, desc='Processing views in internal dataset'):
                # print(f"View: {view_name}")  # Debug statement
                logger.info(f"Processing view: {view_name}")
                extract_features_internal(view_name, video_path, anno_path, output_path, feature_type, preprocess, model)


def extract_features_external(filename, video_path, anno_path, output_path, feature_type, preprocess, model):
    """Extract features for external dataset."""
    filename = filename[:filename.rfind('.')]  # e.g., vid.mp4 -> vid
    video_full_path = os.path.join(video_path, filename + '.mp4')  # Full path to the video file
    
    # Debug: Check if video file exists and can be opened
    if not os.path.exists(video_full_path):
        print(f"Error: Video file does not exist: {video_full_path}")
        return
    cap = cv.VideoCapture(video_full_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Failed to open video: {video_full_path}")
        return

    print(f"Successfully opened video: {video_full_path}")
    
    # Skip annotation loading if feature_type is 'global' and no annotations are needed
    if feature_type == 'global' and not anno_path:
        bbox_to_cut = [(0, 0, 1000, 1000)]  # Use full frame (no bbox)
    else:
        # Proceed as normal if annotations are required
        bbox_full_path = os.path.join(anno_path, filename + '_bbox.json')
        try:
            bbox_to_cut = get_square_box(video_full_path, bbox_full_path, feature_type)
        except Exception as e:
            print(f"Error processing bounding boxes: {e}")
            bbox_to_cut = []

    imfeat = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {frame_count}, exiting...")
            break
        if frame_count % frame_skip == 0:
            # Debug: Check if frame is valid and print its shape
            print(f"Processing frame {frame_count}, size: {frame.shape}")
    
            # Process the entire frame for global/sub_global
            if feature_type in ['global', 'sub_global']:
                # For global or sub-global, we use the entire frame
                x1, y1, x2, y2 = bbox_to_cut[-1]  # Use full frame for global/sub-global
                features = process_frame(frame, (x1, y1, x2, y2), preprocess, model)
                if features is not None:
                    imfeat.append(features)  # Add features to the list
    
        frame_count += 1

    cap.release()

    # Check if features were successfully extracted
    if len(imfeat) == 0:
        print("No features extracted!")
    else:
        print(f"imfeat type before concatenation: {type(imfeat)}")
        print(f"imfeat length before concatenation: {len(imfeat)}")

        if len(imfeat) > 0 and isinstance(imfeat[0], np.ndarray):  # Check if features are valid numpy arrays
            imfeat = np.concatenate(imfeat, axis=0)  # Concatenate all frames' features into one array
            print(f"imfeat shape after concatenation: {imfeat.shape}")
            np.save(os.path.join(output_path, filename + '.npy'), imfeat)  # Save the concatenated features
        else:
            print("Error: imfeat contains invalid data. Expected numpy arrays.")



# Extract features per view
def extract_features_view(view, view_name, view_path, anno_path, output_path, feature_type, preprocess, model):
    """Extract features for each view."""
    
    # If you're processing a single video, remove the loop over view files
    video_full_path = os.path.join(view_path, view_name + '.mp4')  # Full path to the single video file

    # Process the single video file directly
    imfeat = []  # Initialize to store features for global/sub_global extraction
    cap = cv.VideoCapture(video_full_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame, exiting...")
            break  # Exit if no more frames are available

        # Process the entire frame for global/sub_global
        if feature_type in ['global', 'sub_global']:
            # For global or sub-global, we use the entire frame
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]  # Use full frame coordinates
            features = process_frame(frame, (x1, y1, x2, y2), preprocess, model)
            if features is not None:
                imfeat.append(features)  # Add features to the list

        frame_count += 1

    cap.release()

    # Save the output for global and sub_global feature extraction
    if feature_type != 'local':
        imfeat = np.concatenate(imfeat, axis=0)
        np.save(os.path.join(output_path, view_name + '.npy'), imfeat)

@logger.catch
def main(config_path, is_external):
    """
    CLI interface to extract features from videos using CLIP model.
    Args:
        config_path (str): Path to the configuration YAML file.
        is_external (bool): Whether the dataset is external or not.
    Returns:
        None
    """
    try:
        configs = load_configs(config_path)
        extract_features(configs, is_external)
        print("Feature extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Using Fire to handle the CLI
if __name__ == "__main__":
    fire.Fire(main)
