import numpy as np
from pathlib import Path
import os
from ultralytics import YOLO
from PIL import Image

def next_roi(roi_data, adc_line):
    adc_line = adc_line.split(",")
    roi_x = int(adc_line[15])  # ROI width
    roi_y = int(adc_line[16])  # ROI height
    # Skip empty roi
    if roi_x < 1 or roi_y < 1:
        return None
    # roi_data is a 1-dimensional array, where
    # all roi are stacked one after another.
    start = int(adc_line[17])  # start byte
    end = start + (roi_x * roi_y)
    # Reshape into 2-dimensions
    return roi_data[start:end].reshape((roi_y, roi_x))

def raw_to_numpy(adc, roi):
    adc = Path(adc)
    # Read bytes from .roi-file into 8-bit integers
    roi_data = np.fromfile(roi, dtype="uint8")
    # Parse each line of .adc-file
    with adc.open() as adc_fh:
        for i, adc_line in enumerate(adc_fh, start=1):
            np_arr = next_roi(roi_data, adc_line)
            if np_arr is not None:
                yield i, np_arr

def classify_ifcb_in_folder(model_path, input_folder_path, output_folder_path = 'results', confidence_threshold = 0.5):
    model = YOLO(model_path)

    for file in Path(input_folder_path).glob("*.roi"):

        # Get base file name
        base_file_name = file.stem
        roi_path = file
        adc_path = file.with_suffix(".adc")

        # Read image as numpy array
        all_img = raw_to_numpy(adc_path, roi_path)

        for index, roi_array in all_img:
            results = model.predict(roi_array)

            predicted_name = results[0].names[results[0].probs.top1]
            predicted_conf = results[0].probs.top1conf
            predicted_conf = np.round(predicted_conf, 2)

            if predicted_conf > confidence_threshold:
        
                array = np.clip(roi_array, 0, 255).astype(np.uint8)
    
                # Create image from array
                img = Image.fromarray(array)

                # Set output folder
                save_folder_path = os.path.join(output_folder_path, predicted_name)
                os.makedirs(save_folder_path, exist_ok=True)

                # Set filename
                save_file_name = f"{base_file_name}_{index}_{predicted_conf}.png"

                # Save image
                img.save(os.path.join(save_folder_path, save_file_name))