import os
import glob
import yaml
import shutil
from pathlib import Path
import re

# Rename the existing function to is_yolo11_dataset
def is_yolo11_dataset(folder_path):
    """
    Check if the given folder follows the structure of a YOLO11 dataset.
    
    Structure of YOLO11 dataset:
    dataset/
       ├── dataset.yaml         # configuration file
       ├── train/
       │   ├── images/       # directory with images for train subset
       │   └── labels/       # directory with labels for train subset
       ├── valid/
       │   ├── images/       # directory with images for validation subset
       │   └── labels/       # directory with labels for validation subset
       └── test/             # optional
           ├── images/       # directory with images for test subset
           └── labels/       # directory with labels for test subset
    
    Args:
        folder_path (str): Path to the folder to check
        
    Returns:
        dict: A dictionary with the validation results and details
    """
    result = {
        "is_valid": False,
        "structure": {},
        "issues": []
    }
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        result["issues"].append(f"Folder path '{folder_path}' does not exist")
        return result
    
    # Check for dataset.yaml configuration file
    yaml_path = os.path.join(folder_path, "dataset.yaml")
    if not os.path.isfile(yaml_path):
        result["issues"].append("dataset.yaml configuration file not found")
    else:
        result["structure"]["data_yaml"] = True
        
        # Validate YAML structure
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                
                # Check for expected YAML keys
                expected_keys = ["train", "val", "names", "nc"]
                found_keys = [key for key in expected_keys if key in yaml_content]
                result["structure"]["config_keys"] = found_keys
                
                # Check if required keys are present
                missing_keys = [key for key in ["train", "names"] if key not in found_keys]
                if missing_keys:
                    result["issues"].append(f"Missing required keys in dataset.yaml: {', '.join(missing_keys)}")
                
                # Check if class names are defined
                if "names" in yaml_content:
                    if isinstance(yaml_content["names"], dict):
                        result["structure"]["classes"] = list(yaml_content["names"].values())
                        result["structure"]["class_count"] = len(yaml_content["names"])
                    elif isinstance(yaml_content["names"], list):
                        result["structure"]["classes"] = yaml_content["names"]
                        result["structure"]["class_count"] = len(yaml_content["names"])
                        
                        # Check if nc matches the number of classes
                        if "nc" in yaml_content and yaml_content["nc"] != len(yaml_content["names"]):
                            result["issues"].append(f"Number of classes (nc: {yaml_content['nc']}) doesn't match names list length ({len(yaml_content['names'])})")
                    else:
                        result["issues"].append("Invalid 'names' field in dataset.yaml (should be a dictionary or list)")
                else:
                    result["issues"].append("Missing 'names' field in dataset.yaml")
        except Exception as e:
            result["issues"].append(f"Error parsing YAML config: {str(e)}")
    
    # Check for train/valid/test splits and their subfolders
    required_splits = ["train"]
    optional_splits = ["valid", "test"]
    
    for split in required_splits + optional_splits:
        # Use "valid" for both "val" and "valid" checks
        split_dir = os.path.join(folder_path, "valid" if split == "val" else split)
        is_required = split in required_splits
        
        if os.path.isdir(split_dir):
            result["structure"][f"{split}_dir"] = True
            
            # Check for images subfolder
            images_dir = os.path.join(split_dir, "images")
            if os.path.isdir(images_dir):
                result["structure"][f"{split}_images_dir"] = True
                
                # Count image files
                image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
                image_files.extend(glob.glob(os.path.join(images_dir, "*.jpeg")))
                image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
                image_files.extend(glob.glob(os.path.join(images_dir, "*.bmp")))
                
                if image_files:
                    result["structure"][f"{split}_image_count"] = len(image_files)
                    result["structure"][f"{split}_sample_image"] = os.path.basename(image_files[0])
                else:
                    message = f"No image files found in {split}/images/"
                    if is_required:
                        result["issues"].append(message)
                    else:
                        result["structure"][f"{split}_warning"] = message
            else:
                message = f"'{split}/images' directory not found"
                if is_required:
                    result["issues"].append(message)
                else:
                    result["structure"][f"{split}_warning"] = message
            
            # Check for labels subfolder
            labels_dir = os.path.join(split_dir, "labels")
            if os.path.isdir(labels_dir):
                result["structure"][f"{split}_labels_dir"] = True
                
                # Count label files
                label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
                
                if label_files:
                    result["structure"][f"{split}_label_count"] = len(label_files)
                    result["structure"][f"{split}_sample_label"] = os.path.basename(label_files[0])
                    
                    # Check if images have corresponding labels and vice versa
                    if f"{split}_image_count" in result["structure"]:
                        image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
                        label_basenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
                        
                        missing_labels = set(image_basenames) - set(label_basenames)
                        if missing_labels:
                            result["issues"].append(f"Some {split} images missing label files: {len(missing_labels)} files")
                            
                        extra_labels = set(label_basenames) - set(image_basenames)
                        if extra_labels:
                            result["issues"].append(f"Some {split} label files have no corresponding images: {len(extra_labels)} files")
                    
                    # Validate label format (check first file)
                    try:
                        with open(label_files[0], 'r') as f:
                            label_content = f.read().strip()
                            if label_content:
                                lines = label_content.split('\n')
                                first_line = lines[0].split()
                                
                                # Check YOLO format: class_id x_center y_center width height
                                if len(first_line) != 5:
                                    result["issues"].append(f"Label file does not follow YOLO format (class_id cx cy width height)")
                                else:
                                    try:
                                        # Check if values are numeric and in valid range (0-1 for coordinates)
                                        class_id = int(first_line[0])
                                        coords = [float(val) for val in first_line[1:5]]
                                        valid_coords = all(0 <= val <= 1 for val in coords)
                                        
                                        if not valid_coords:
                                            result["issues"].append(f"Label coordinates are not normalized (should be between 0-1)")
                                        else:
                                            result["structure"]["valid_label_format"] = True
                                    except ValueError:
                                        result["issues"].append(f"Label values are not numeric")
                    except Exception as e:
                        result["issues"].append(f"Error reading label file: {str(e)}")
                else:
                    message = f"No label files found in {split}/labels/"
                    if is_required:
                        result["issues"].append(message)
                    else:
                        result["structure"][f"{split}_warning"] = message
            else:
                message = f"'{split}/labels' directory not found"
                if is_required:
                    result["issues"].append(message)
                else:
                    result["structure"][f"{split}_warning"] = message
        elif is_required:
            result["issues"].append(f"Required '{split}' directory not found")
    
    # Final verdict - check minimal requirements for a valid YOLO11 dataset
    # 1. dataset.yaml with at least train path and class names
    # 2. train directory with images and labels
    # 3. Valid YOLO label format
    critical_checks = [
        "data_yaml" in result["structure"],
        "train_dir" in result["structure"],
        "train_images_dir" in result["structure"],
        "train_labels_dir" in result["structure"],
        "valid_label_format" in result["structure"]
    ]
    
    # Add check for non-empty directories if they exist
    if "train_images_dir" in result["structure"]:
        critical_checks.append("train_image_count" in result["structure"] and result["structure"]["train_image_count"] > 0)
    
    if "train_labels_dir" in result["structure"]:
        critical_checks.append("train_label_count" in result["structure"] and result["structure"]["train_label_count"] > 0)
    
    result["is_valid"] = all(critical_checks)
    
    return result

# Use the existing is_yolov8_dataset function
def is_yolov8_dataset(folder_path, require_val=False, require_test=False):
    """
    Check if the given folder follows the structure of an Ultralytics YOLOv8 dataset.
    
    Args:
        folder_path (str): Path to the dataset root folder
        require_val (bool): Whether to require validation set
        require_test (bool): Whether to require test set
        
    Returns:
        dict: A dictionary with the validation results and details
    """
    result = {
        "is_valid": False,
        "structure": {},
        "issues": []
    }
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        result["issues"].append(f"Dataset path '{folder_path}' does not exist")
        return result
    
    # Check for dataset.yaml configuration file
    yaml_path = os.path.join(folder_path, "dataset.yaml")
    if not os.path.isfile(yaml_path):
        result["issues"].append("dataset.yaml configuration file not found")
    else:
        result["structure"]["data_yaml"] = True
        
        # Validate YAML structure
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                
                # Check for required YAML keys
                required_keys = ["path", "train", "names"]
                missing_keys = [key for key in required_keys if key not in yaml_content]
                
                if missing_keys:
                    result["issues"].append(f"dataset.yaml missing required keys: {', '.join(missing_keys)}")
                else:
                    result["structure"]["yaml_keys"] = list(yaml_content.keys())
                
                # Check if class names are defined correctly
                if "names" in yaml_content:
                    if isinstance(yaml_content["names"], dict):
                        result["structure"]["classes"] = list(yaml_content["names"].values())
                        result["structure"]["class_count"] = len(yaml_content["names"])
                    elif isinstance(yaml_content["names"], list):
                        result["structure"]["classes"] = yaml_content["names"]
                        result["structure"]["class_count"] = len(yaml_content["names"])
                        
                        # Check if nc matches the number of classes
                        if "nc" in yaml_content and yaml_content["nc"] != len(yaml_content["names"]):
                            result["issues"].append(f"Number of classes (nc: {yaml_content['nc']}) doesn't match names list length ({len(yaml_content['names'])})")
                    else:
                        result["issues"].append("Invalid 'names' field in dataset.yaml (should be a dictionary or list)")
                else:
                    result["issues"].append("Missing 'names' field in dataset.yaml")
                
                # Validate paths in YAML
                if "train" in yaml_content:
                    train_txt = yaml_content["train"]
                    if not isinstance(train_txt, str):
                        result["issues"].append(f"'train' in dataset.yaml should be a string, got {type(train_txt).__name__}")
                
                if "val" in yaml_content:
                    result["structure"]["has_val_in_yaml"] = True
                
                if "test" in yaml_content:
                    result["structure"]["has_test_in_yaml"] = True
                    
        except Exception as e:
            result["issues"].append(f"Error parsing dataset.yaml: {str(e)}")
    
    # Check for subset text files (train.txt, val.txt, test.txt)
    for subset in ["train", "val", "test"]:
        is_required = (subset == "train" or 
                      (subset == "val" and require_val) or 
                      (subset == "test" and require_test))
        
        txt_path = os.path.join(folder_path, f"{subset}.txt")
        if os.path.isfile(txt_path):
            result["structure"][f"{subset}_txt"] = True
            
            # Check content of the text file
            try:
                with open(txt_path, 'r') as f:
                    image_paths = [line.strip() for line in f.readlines() if line.strip()]
                    result["structure"][f"{subset}_image_count"] = len(image_paths)
                    
                    if len(image_paths) == 0:
                        result["issues"].append(f"{subset}.txt is empty")
                    else:
                        # Check if the paths follow the expected format
                        valid_paths = all(re.match(r'^images/.+\.(jpg|jpeg|png|bmp)$', path, re.IGNORECASE) for path in image_paths)
                        if not valid_paths:
                            result["issues"].append(f"Some paths in {subset}.txt do not follow 'images/<subset>/<filename>' format")
                        
                        # Check if the image files exist
                        missing_images = []
                        for path in image_paths[:10]:  # Check just the first 10 to avoid excessive checking
                            full_path = os.path.join(folder_path, path)
                            if not os.path.isfile(full_path):
                                missing_images.append(path)
                        
                        if missing_images:
                            result["issues"].append(f"Some image files listed in {subset}.txt do not exist: {', '.join(missing_images[:3])}...")
            except Exception as e:
                result["issues"].append(f"Error reading {subset}.txt: {str(e)}")
        elif is_required:
            result["issues"].append(f"Required {subset}.txt file not found")
    
    # Check folder structure
    images_dir = os.path.join(folder_path, "images")
    labels_dir = os.path.join(folder_path, "labels")
    
    if not os.path.isdir(images_dir):
        result["issues"].append("'images' directory not found")
    else:
        result["structure"]["images_dir"] = True
    
    if not os.path.isdir(labels_dir):
        result["issues"].append("'labels' directory not found")
    else:
        result["structure"]["labels_dir"] = True
    
    # Check for train/val/test subfolders and their contents
    for subset in ["train", "val", "test"]:
        is_required = (subset == "train" or 
                      (subset == "val" and require_val) or 
                      (subset == "test" and require_test))
        
        images_subset_dir = os.path.join(images_dir, subset)
        labels_subset_dir = os.path.join(labels_dir, subset)
        
        # Check images subfolder
        if os.path.isdir(images_subset_dir):
            result["structure"][f"{subset}_images_dir"] = True
            
            # Count image files
            image_files = glob.glob(os.path.join(images_subset_dir, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.jpeg")))
            image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.png")))
            image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.bmp")))
            
            result["structure"][f"{subset}_image_files"] = len(image_files)
            
            if len(image_files) == 0 and is_required:
                result["issues"].append(f"No image files found in images/{subset}/")
        elif is_required:
            result["issues"].append(f"Required 'images/{subset}' directory not found")
        
        # Check labels subfolder
        if os.path.isdir(labels_subset_dir):
            result["structure"][f"{subset}_labels_dir"] = True
            
            # Count label files
            label_files = glob.glob(os.path.join(labels_subset_dir, "*.txt"))
            result["structure"][f"{subset}_label_files"] = len(label_files)
            
            if len(label_files) == 0 and is_required:
                result["issues"].append(f"No label files found in labels/{subset}/")
            elif len(label_files) > 0:
                # Check if image files have corresponding label files and vice versa
                if os.path.isdir(images_subset_dir) and len(image_files) > 0:
                    # Handle complex filenames with multiple dots
                    image_basenames = []
                    for img_path in image_files:
                        img_filename = os.path.basename(img_path)
                        # Get everything before the last extension (.jpg, .png, etc.)
                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                            if img_filename.endswith(ext):
                                img_basename = img_filename[:-len(ext)]
                                image_basenames.append(img_basename)
                                break
                        else:
                            # If no recognized extension is found, use the whole filename
                            image_basenames.append(img_filename)
                    
                    # Do the same for label files
                    label_basenames = []
                    for lbl_path in label_files:
                        lbl_filename = os.path.basename(lbl_path)
                        if lbl_filename.endswith('.txt'):
                            label_basenames.append(lbl_filename[:-4])  # Remove .txt
                        else:
                            label_basenames.append(lbl_filename)
                    
                    missing_labels = set(image_basenames) - set(label_basenames)
                    if missing_labels and is_required:
                        result["issues"].append(f"Some {subset} images are missing label files: {len(missing_labels)} total")
                        
                    extra_labels = set(label_basenames) - set(image_basenames)
                    if extra_labels:
                        result["issues"].append(f"Some {subset} label files have no corresponding images: {len(extra_labels)} total")
                
                # Validate label file format (check first file)
                try:
                    with open(label_files[0], 'r') as f:
                        label_content = f.read().strip()
                        if label_content:
                            lines = label_content.split('\n')
                            first_line = lines[0].split()
                            
                            # Check YOLO format: class_id cx cy width height
                            if len(first_line) != 5:
                                result["issues"].append(f"Label file does not follow YOLO format (class_id cx cy width height)")
                            else:
                                try:
                                    # Check if values are numeric and in valid range (0-1 for coordinates)
                                    class_id = int(first_line[0])
                                    coords = [float(val) for val in first_line[1:5]]
                                    valid_coords = all(0 <= val <= 1 for val in coords)
                                    
                                    if not valid_coords:
                                        result["issues"].append(f"Label coordinates are not normalized (should be between 0-1)")
                                    else:
                                        result["structure"]["valid_label_format"] = True
                                except ValueError:
                                    result["issues"].append(f"Label values are not numeric")
                except Exception as e:
                    result["issues"].append(f"Error reading label file: {str(e)}")
        elif is_required:
            result["issues"].append(f"Required 'labels/{subset}' directory not found")
    
    # Final verdict
    # A valid YOLOv8 dataset must have:
    # 1. Valid dataset.yaml with proper keys (path, train, names)
    # 2. train.txt (and val.txt/test.txt if required)
    # 3. images/train/ directory with image files
    # 4. labels/train/ directory with label files in correct format
    # 5. Matching image and label files
    
    # Create a list of critical checks
    critical_checks = [
        "data_yaml" in result["structure"],
        "train_txt" in result["structure"] or not is_required,
        "train_images_dir" in result["structure"] or not is_required,
        "train_labels_dir" in result["structure"] or not is_required,
        ("train_image_files" in result["structure"] and result["structure"]["train_image_files"] > 0) or not is_required,
        ("train_label_files" in result["structure"] and result["structure"]["train_label_files"] > 0) or not is_required,
        "valid_label_format" in result["structure"],
        ("val_txt" in result["structure"]) or not require_val,
        ("test_txt" in result["structure"]) or not require_test
    ]
    
    result["is_valid"] = all(critical_checks)
    
    return result