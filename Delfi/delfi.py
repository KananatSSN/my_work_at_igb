import os
from ultralytics import YOLO
import random
import datetime
import yaml
import shutil
import glob

'''
The DatasetBuilder is a class that helps to create a YOLOv11 dataset from raw [Delfi data] and a model. It also check if the data in the [raw_dataset] is already in the [unverified_dataset] or not before processing.
The intended use case is to use this function to build a (bad) dataset from model predictions, correct the dataset with annotation tool (like Roboflow or CVAT) and then use the corrected dataset to retrain the model.

Definition
Delfi data = The .jpg images from the Delfi imager.
raw_dataset = The folder containing the Delfi data. (This script will look for .jpg files in this folder and all it's subfolders)
unverified_dataset = The dataset that is need to be manually corrected. (The output from this script)
'''

class DatasetBuilder:
    def __init__(self, raw_dataset_path, unverified_dataset_path):
        self.raw_dataset_path = raw_dataset_path
        self.unverified_dataset_path = unverified_dataset_path
        self.unprocessed_raw_images = self.find_unprocessed_raw_data()

    def create_yolo11_dataset_with_model(self, model_path, conf_threshold = 0.2, n_images = 10, split_ratio = [0.7, 0.2, 0.1], verbose = False):

        # Check if we have enough unprocessed images
        if n_images > len(self.unprocessed_raw_images):

            n_images = len(self.unprocessed_raw_images)
            if n_images == 0:
                print("No unprocessed images found")
                return
            
            print(f"Number of images to process is greater than the number of unprocessed images. Setting n_images to {n_images} (all images)")
        
        # Load the model
        model = YOLO(model_path)

        '''
        Create Yolo dataset structure
        '''

        # Create a folder to store the unverified images
        now = datetime.datetime.now()
        date = now.strftime("%d%m%Y")
        time = now.strftime("%H%M%S")

        output_dataset_name = f"Unverified_{date}_{time}"
        output_dataset_path = os.path.join(self.unverified_dataset_path, output_dataset_name)

        dataset_structure = self.create_yolo11_dataset_structure(output_dataset_path, model.names)

        '''
        Process the images
        '''

        # Randomly select n_images from the unprocessed images
        image_to_process = random.sample(self.unprocessed_raw_images, n_images)

        if verbose:
            raw_images_list = self.find_jpg_files(self.raw_dataset_path)
            print(f"Total raw images: {len(raw_images_list)}")
            print(f"Unprocessed raw images: {len(self.unprocessed_raw_images)}")
            print(f"Images to process: {len(image_to_process)} = {image_to_process}")

        i = 0

        for image_path in image_to_process:

            i += 1

            if verbose:
                print(f"Processing {i} out of {len(image_to_process)} : {image_path}")
            
            # Get the image name
            image_name = os.path.basename(image_path).split('.')[0]

            # Generate a random number to decide whether to save the image to train, val, or test
            random_number = random.random()
            if random_number < split_ratio[0]:
                output_images_folder = dataset_structure['train']['images']
                output_labels_folder = dataset_structure['train']['labels']
            elif random_number < split_ratio[0] + split_ratio[1]:
                output_images_folder = dataset_structure['val']['images']
                output_labels_folder = dataset_structure['val']['labels']
            else:
                output_images_folder = dataset_structure['test']['images']
                output_labels_folder = dataset_structure['test']['labels']

            # Run inference on the image
            results = model(image_path, conf=conf_threshold, verbose=False)

            # Save image to the output images folder
            shutil.copy(image_path, os.path.join(output_images_folder, f"{image_name}.jpg"))

            # Save label to the output labels folder
            for result in results:
                result.save_txt(os.path.join(output_labels_folder, f"{image_name}.txt"), save_conf=False)

            # Remvoe the image from the unprocessed list
            self.unprocessed_raw_images.remove(image_path)

    def find_unprocessed_raw_data(self):

        raw_images_list = self.find_jpg_files(self.raw_dataset_path)
        unverified_images_list = self.find_jpg_files(self.unverified_dataset_path)

        # Extract just the filenames (not full paths) for comparison
        raw_filenames = {os.path.basename(path) for path in raw_images_list}
        unverified_filenames = {os.path.basename(path) for path in unverified_images_list}

        # Files that have been processed are in either unverified or verified
        unprocessed_filenames = list(raw_filenames - unverified_filenames)

        unprocessed_images = [path for path in raw_images_list if os.path.basename(path) in unprocessed_filenames]

        return unprocessed_images
    

    @staticmethod
    def create_yolo11_dataset_structure(dataset_path, class_name):
        # Create train, val, test folders
        train_images_folder_path = os.path.join(dataset_path, "train", "images")
        train_labels_folder_path = os.path.join(dataset_path, "train", "labels")
        val_images_folder_path = os.path.join(dataset_path, "val", "images")
        val_labels_folder_path = os.path.join(dataset_path, "val", "labels")
        test_images_folder_path = os.path.join(dataset_path, "test", "images")
        test_labels_folder_path = os.path.join(dataset_path, "test", "labels")

        os.makedirs(train_images_folder_path, exist_ok=True)
        os.makedirs(train_labels_folder_path, exist_ok=True)
        os.makedirs(val_images_folder_path, exist_ok=True)
        os.makedirs(val_labels_folder_path, exist_ok=True)
        os.makedirs(test_images_folder_path, exist_ok=True)
        os.makedirs(test_labels_folder_path, exist_ok=True)

        dataset_structure = {
            'train': {'images': train_images_folder_path, 'labels': train_labels_folder_path},
            'val': {'images': val_images_folder_path, 'labels': val_labels_folder_path},
            'test': {'images': test_images_folder_path, 'labels': test_labels_folder_path}
        }

        # Create dataset.yaml
        yaml_content = {
            'train': str("./train/images"),
            'val': str("./val/images"),
            'test': str("./test/images"),
            'nc': len(class_name),
            'names': class_name
        }

        with open(os.path.join(dataset_path, "dataset.yaml"), 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        return dataset_structure

    @staticmethod
    def find_jpg_files(directory):
        """
        Search through all files and folders in the given directory
        and return a list of paths to all .jpg files, excluding those with 
        variations of 'sort' at the end of the filename.
        
        Args:
            directory (str): The root directory to search in
            
        Returns:
            list: List of full paths to all .jpg files found (excluding sort variants)
        """
        jpg_files_list = []
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file has .jpg extension (case-insensitive)
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    jpg_files_list.append(full_path)
        
        return jpg_files_list

'''
The DatasetConverter is supposed to convert [Ultralytics yolov11 dataset] to [CVAT integration of Ultralytics yolov8 dataset], but this is not working.
For more information
This is the Yolov11 dataset format : https://docs.ultralytics.com/datasets/detect/ (If you use the DatasetBuilder, this is the format)
This is CVAT integration of ultralytics Yolo format : https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/
'''

# class DatasetConverter:

#     def __init__(self, original_dataset_path, converted_dataset_path):
#         self.original_dataset_path = original_dataset_path
#         self.converted_dataset_path = converted_dataset_path

#         # Check if the original dataset is in YOLO11 format
#         self.check_if_yolo11 = self.is_yolo11_dataset(self.original_dataset_path)
#         if not self.check_if_yolo11["is_valid"]:
#             print("Dataset is not in YOLO11 format")
#             print(self.check_if_yolo11["issues"])
#             return

#     # def convert_dataset_yolo11_to_yolo8(self):

#     @staticmethod
#     def is_yolo11_dataset(folder_path):
#         """
#         Check if the given folder follows the structure of a YOLO dataset.
        
#         Args:
#             folder_path (str): Path to the folder to check
            
#         Returns:
#             dict: A dictionary with the validation results and details
#         """
#         result = {
#             "is_valid": False,
#             "structure": {},
#             "issues": []
#         }
        
#         # Check if the folder exists
#         if not os.path.isdir(folder_path):
#             result["issues"].append(f"Folder path '{folder_path}' does not exist")
#             return result
        
#         # Check for images folder
#         images_folder = None
#         for img_dir in ["images", "img", "JPEGImages"]:
#             if os.path.isdir(os.path.join(folder_path, img_dir)):
#                 images_folder = os.path.join(folder_path, img_dir)
#                 break
        
#         if images_folder:
#             result["structure"]["images_folder"] = images_folder
#             # Check for image files
#             image_files = glob.glob(os.path.join(images_folder, "**/*.jpg"), recursive=True)
#             image_files.extend(glob.glob(os.path.join(images_folder, "**/*.jpeg"), recursive=True))
#             image_files.extend(glob.glob(os.path.join(images_folder, "**/*.png"), recursive=True))
            
#             if not image_files:
#                 result["issues"].append("No image files found in the images folder")
#             else:
#                 result["structure"]["image_count"] = len(image_files)
#                 result["structure"]["sample_image"] = os.path.basename(image_files[0])
#         else:
#             result["issues"].append("No images folder found")
        
#         # Check for labels folder
#         labels_folder = None
#         for lbl_dir in ["labels", "txt", "annotations"]:
#             if os.path.isdir(os.path.join(folder_path, lbl_dir)):
#                 labels_folder = os.path.join(folder_path, lbl_dir)
#                 break
        
#         if labels_folder:
#             result["structure"]["labels_folder"] = labels_folder
#             # Check for label files
#             label_files = glob.glob(os.path.join(labels_folder, "**/*.txt"), recursive=True)
            
#             if not label_files:
#                 result["issues"].append("No label files (*.txt) found in the labels folder")
#             else:
#                 result["structure"]["label_count"] = len(label_files)
#                 result["structure"]["sample_label"] = os.path.basename(label_files[0])
                
#                 # Validate label format (check first file)
#                 try:
#                     with open(label_files[0], 'r') as f:
#                         label_content = f.read().strip()
#                         if label_content:
#                             lines = label_content.split('\n')
#                             first_line = lines[0].split()
                            
#                             # Check YOLO format: class_id x_center y_center width height
#                             if len(first_line) != 5:
#                                 result["issues"].append("Label file does not follow YOLO format (class_id x y width height)")
#                             else:
#                                 try:
#                                     # Check if values are numeric and in valid range (0-1 for coordinates)
#                                     class_id = int(first_line[0])
#                                     coords = [float(val) for val in first_line[1:5]]
#                                     valid_coords = all(0 <= val <= 1 for val in coords)
                                    
#                                     if not valid_coords:
#                                         result["issues"].append("Label coordinates are not normalized (should be between 0-1)")
#                                     else:
#                                         result["structure"]["valid_label_format"] = True
#                                 except ValueError:
#                                     result["issues"].append("Label values are not numeric")
#                 except Exception as e:
#                     result["issues"].append(f"Error reading label file: {str(e)}")
#         else:
#             result["issues"].append("No labels folder found")
        
#         # Check for YAML configuration file
#         yaml_files = glob.glob(os.path.join(folder_path, "*.yaml"))
#         yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yml")))
        
#         if yaml_files:
#             result["structure"]["config_file"] = os.path.basename(yaml_files[0])
            
#             # Validate YAML structure
#             try:
#                 with open(yaml_files[0], 'r') as f:
#                     yaml_content = yaml.safe_load(f)
                    
#                     # Check for typical YOLO dataset YAML keys
#                     expected_keys = ["path", "train", "val", "names"]
#                     found_keys = [key for key in expected_keys if key in yaml_content]
#                     result["structure"]["config_keys"] = found_keys
                    
#                     # Check if class names are defined
#                     if "names" in yaml_content and isinstance(yaml_content["names"], dict):
#                         result["structure"]["classes"] = list(yaml_content["names"].values())
#             except Exception as e:
#                 result["issues"].append(f"Error parsing YAML config: {str(e)}")
#         else:
#             # Check for classes.txt or classes.names as alternative
#             class_files = glob.glob(os.path.join(folder_path, "*.names"))
#             class_files.extend(glob.glob(os.path.join(folder_path, "classes.txt")))
            
#             if class_files:
#                 result["structure"]["classes_file"] = os.path.basename(class_files[0])
                
#                 # Read class names
#                 try:
#                     with open(class_files[0], 'r') as f:
#                         classes = [line.strip() for line in f.readlines() if line.strip()]
#                         result["structure"]["classes"] = classes
#                 except Exception as e:
#                     result["issues"].append(f"Error reading classes file: {str(e)}")
#             else:
#                 result["issues"].append("No dataset configuration (YAML) or classes file found")
        
#         # Check for train/val/test splits
#         for split in ["train", "val", "test"]:
#             split_dir = os.path.join(folder_path, split)
#             if os.path.isdir(split_dir):
#                 result["structure"][f"{split}_split"] = True
        
#         # Final verdict
#         if (
#             "images_folder" in result["structure"] and 
#             "labels_folder" in result["structure"] and 
#             ("config_file" in result["structure"] or "classes_file" in result["structure"]) and
#             ("image_count" in result["structure"] and result["structure"]["image_count"] > 0) and
#             ("label_count" in result["structure"] and result["structure"]["label_count"] > 0)
#         ):
#             result["is_valid"] = True
        
#         return result
    
#     def is_yolov8_dataset(folder_path, require_val=False, require_test=False):
#         """
#         Check if the given folder follows the structure of an Ultralytics YOLOv8 dataset.
        
#         Args:
#             folder_path (str): Path to the dataset root folder
#             require_val (bool): Whether to require validation set
#             require_test (bool): Whether to require test set
            
#         Returns:
#             dict: A dictionary with the validation results and details
#         """
#         result = {
#             "is_valid": False,
#             "structure": {},
#             "issues": []
#         }
        
#         # Check if the folder exists
#         if not os.path.isdir(folder_path):
#             result["issues"].append(f"Dataset path '{folder_path}' does not exist")
#             return result
        
#         # Check for data.yaml configuration file
#         yaml_path = os.path.join(folder_path, "data.yaml")
#         if not os.path.isfile(yaml_path):
#             result["issues"].append("data.yaml configuration file not found")
#         else:
#             result["structure"]["data_yaml"] = True
            
#             # Validate YAML structure
#             try:
#                 with open(yaml_path, 'r') as f:
#                     yaml_content = yaml.safe_load(f)
                    
#                     # Check for required YAML keys
#                     required_keys = ["path", "train", "names"]
#                     missing_keys = [key for key in required_keys if key not in yaml_content]
                    
#                     if missing_keys:
#                         result["issues"].append(f"data.yaml missing required keys: {', '.join(missing_keys)}")
#                     else:
#                         result["structure"]["yaml_keys"] = list(yaml_content.keys())
                    
#                     # Check if class names are defined correctly
#                     if "names" in yaml_content and isinstance(yaml_content["names"], dict):
#                         result["structure"]["classes"] = list(yaml_content["names"].values())
#                         result["structure"]["class_count"] = len(yaml_content["names"])
#                     else:
#                         result["issues"].append("Invalid 'names' field in data.yaml (should be a dictionary)")
                    
#                     # Validate paths in YAML
#                     if "train" in yaml_content:
#                         train_txt = yaml_content["train"]
#                         if not isinstance(train_txt, str):
#                             result["issues"].append(f"'train' in data.yaml should be a string, got {type(train_txt).__name__}")
                    
#                     if "val" in yaml_content:
#                         result["structure"]["has_val_in_yaml"] = True
                    
#                     if "test" in yaml_content:
#                         result["structure"]["has_test_in_yaml"] = True
                        
#             except Exception as e:
#                 result["issues"].append(f"Error parsing data.yaml: {str(e)}")
        
#         # Check for subset text files (train.txt, val.txt, test.txt)
#         for subset in ["train", "val", "test"]:
#             is_required = (subset == "train" or 
#                         (subset == "val" and require_val) or 
#                         (subset == "test" and require_test))
            
#             txt_path = os.path.join(folder_path, f"{subset}.txt")
#             if os.path.isfile(txt_path):
#                 result["structure"][f"{subset}_txt"] = True
                
#                 # Check content of the text file
#                 try:
#                     with open(txt_path, 'r') as f:
#                         image_paths = [line.strip() for line in f.readlines() if line.strip()]
#                         result["structure"][f"{subset}_image_count"] = len(image_paths)
                        
#                         if len(image_paths) == 0:
#                             result["issues"].append(f"{subset}.txt is empty")
#                         else:
#                             # Check if the paths follow the expected format
#                             valid_paths = all(re.match(r'^images/.+\.(jpg|jpeg|png|bmp)$', path, re.IGNORECASE) for path in image_paths)
#                             if not valid_paths:
#                                 result["issues"].append(f"Some paths in {subset}.txt do not follow 'images/<subset>/<filename>' format")
                            
#                             # Check if the image files exist
#                             missing_images = []
#                             for path in image_paths[:10]:  # Check just the first 10 to avoid excessive checking
#                                 full_path = os.path.join(folder_path, path)
#                                 if not os.path.isfile(full_path):
#                                     missing_images.append(path)
                            
#                             if missing_images:
#                                 result["issues"].append(f"Some image files listed in {subset}.txt do not exist: {', '.join(missing_images[:3])}...")
#                 except Exception as e:
#                     result["issues"].append(f"Error reading {subset}.txt: {str(e)}")
#             elif is_required:
#                 result["issues"].append(f"Required {subset}.txt file not found")
        
#         # Check folder structure
#         images_dir = os.path.join(folder_path, "images")
#         labels_dir = os.path.join(folder_path, "labels")
        
#         if not os.path.isdir(images_dir):
#             result["issues"].append("'images' directory not found")
#         else:
#             result["structure"]["images_dir"] = True
        
#         if not os.path.isdir(labels_dir):
#             result["issues"].append("'labels' directory not found")
#         else:
#             result["structure"]["labels_dir"] = True
        
#         # Check for train/val/test subfolders and their contents
#         for subset in ["train", "val", "test"]:
#             is_required = (subset == "train" or 
#                         (subset == "val" and require_val) or 
#                         (subset == "test" and require_test))
            
#             images_subset_dir = os.path.join(images_dir, subset)
#             labels_subset_dir = os.path.join(labels_dir, subset)
            
#             # Check images subfolder
#             if os.path.isdir(images_subset_dir):
#                 result["structure"][f"{subset}_images_dir"] = True
                
#                 # Count image files
#                 image_files = glob.glob(os.path.join(images_subset_dir, "*.jpg"))
#                 image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.jpeg")))
#                 image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.png")))
#                 image_files.extend(glob.glob(os.path.join(images_subset_dir, "*.bmp")))
                
#                 result["structure"][f"{subset}_image_files"] = len(image_files)
                
#                 if len(image_files) == 0 and is_required:
#                     result["issues"].append(f"No image files found in images/{subset}/")
#             elif is_required:
#                 result["issues"].append(f"Required 'images/{subset}' directory not found")
            
#             # Check labels subfolder
#             if os.path.isdir(labels_subset_dir):
#                 result["structure"][f"{subset}_labels_dir"] = True
                
#                 # Count label files
#                 label_files = glob.glob(os.path.join(labels_subset_dir, "*.txt"))
#                 result["structure"][f"{subset}_label_files"] = len(label_files)
                
#                 if len(label_files) == 0 and is_required:
#                     result["issues"].append(f"No label files found in labels/{subset}/")
#                 elif len(label_files) > 0:
#                     # Check if image files have corresponding label files and vice versa
#                     if os.path.isdir(images_subset_dir) and len(image_files) > 0:
#                         image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
#                         label_basenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
                        
#                         missing_labels = set(image_basenames) - set(label_basenames)
#                         if missing_labels and is_required:
#                             result["issues"].append(f"Some {subset} images are missing label files: {', '.join(list(missing_labels)[:3])}...")
                            
#                         extra_labels = set(label_basenames) - set(image_basenames)
#                         if extra_labels:
#                             result["issues"].append(f"Some {subset} label files have no corresponding images: {', '.join(list(extra_labels)[:3])}...")
                    
#                     # Validate label file format (check first file)
#                     try:
#                         with open(label_files[0], 'r') as f:
#                             label_content = f.read().strip()
#                             if label_content:
#                                 lines = label_content.split('\n')
#                                 first_line = lines[0].split()
                                
#                                 # Check YOLO format: class_id cx cy width height
#                                 if len(first_line) != 5:
#                                     result["issues"].append(f"Label file does not follow YOLO format (class_id cx cy width height)")
#                                 else:
#                                     try:
#                                         # Check if values are numeric and in valid range (0-1 for coordinates)
#                                         class_id = int(first_line[0])
#                                         coords = [float(val) for val in first_line[1:5]]
#                                         valid_coords = all(0 <= val <= 1 for val in coords)
                                        
#                                         if not valid_coords:
#                                             result["issues"].append(f"Label coordinates are not normalized (should be between 0-1)")
#                                         else:
#                                             result["structure"]["valid_label_format"] = True
#                                     except ValueError:
#                                         result["issues"].append(f"Label values are not numeric")
#                     except Exception as e:
#                         result["issues"].append(f"Error reading label file: {str(e)}")
#             elif is_required:
#                 result["issues"].append(f"Required 'labels/{subset}' directory not found")
        
#         # Final verdict
#         # A valid YOLOv8 dataset must have:
#         # 1. Valid data.yaml with proper keys (path, train, names)
#         # 2. train.txt (and val.txt/test.txt if required)
#         # 3. images/train/ directory with image files
#         # 4. labels/train/ directory with label files in correct format
#         # 5. Matching image and label files
        
#         # Create a list of critical checks
#         critical_checks = [
#             "data_yaml" in result["structure"],
#             "train_txt" in result["structure"] or not is_required,
#             "train_images_dir" in result["structure"] or not is_required,
#             "train_labels_dir" in result["structure"] or not is_required,
#             ("train_image_files" in result["structure"] and result["structure"]["train_image_files"] > 0) or not is_required,
#             ("train_label_files" in result["structure"] and result["structure"]["train_label_files"] > 0) or not is_required,
#             "valid_label_format" in result["structure"],
#             ("val_txt" in result["structure"]) or not require_val,
#             ("test_txt" in result["structure"]) or not require_test
#         ]
        
#         result["is_valid"] = all(critical_checks)
        
#         return result