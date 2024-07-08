import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Constants
INPUT_DIR = "./COCO_data/train/"
OUTPUT_DIR = "./models/Detectron2_Models/"
TRAIN_JSON = os.path.join(INPUT_DIR, "train/coco_annotations.json")
TRAIN_IMAGES = os.path.join(INPUT_DIR, "train")
VAL_JSON = os.path.join(INPUT_DIR, "val/coco_annotations.json")
VAL_IMAGES = os.path.join(INPUT_DIR, "val")
TEST_JSON = os.path.join(INPUT_DIR, "test/coco_annotations.json")
TEST_IMAGES = os.path.join(INPUT_DIR, "test")
TEST_RESULTS = os.path.join(INPUT_DIR, "test_results")
LABELED_MASKS_DIR = os.path.join(INPUT_DIR, "test_results_labeled_masks")
OUTPUT_CSV = os.path.join(TEST_RESULTS, "output_objects.csv")
MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "model_15k_iter.pth")
CONFIG_YAML = os.path.join(OUTPUT_DIR, "config-5k_iter.yaml")
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

def setup_detectron():
    register_coco_instances("my_dataset_train", {}, TRAIN_JSON, TRAIN_IMAGES)
    register_coco_instances("my_dataset_val", {}, VAL_JSON, VAL_IMAGES)
    register_coco_instances("my_dataset_test", {}, TEST_JSON, TEST_IMAGES)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def load_model(cfg):
    if os.path.exists(MODEL_WEIGHTS):
        print(f"Loading custom model weights from {MODEL_WEIGHTS}")
        cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    else:
        print(f"Custom model weights not found at {MODEL_WEIGHTS}")
        print("Using pre-trained model weights")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    
    try:
        predictor = DefaultPredictor(cfg)
        return predictor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to default model")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return DefaultPredictor(cfg)

def visualize_random_samples(dataset_dicts, metadata, num_samples=2):
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(12, 12))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.show()

def evaluate_model(cfg, predictor):
    evaluator = COCOEvaluator("my_dataset_test", output_dir="./output")
    test_loader = build_detection_test_loader(cfg, "my_dataset_test")
    return inference_on_dataset(predictor.model, test_loader, evaluator)

def process_images(predictor, input_dir, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])
        
        for image_filename in os.listdir(input_dir):
            if not any(image_filename.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                continue
            
            image_path = os.path.join(input_dir, image_filename)
            new_im = cv2.imread(image_path)
            outputs = predictor(new_im)
            
            mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
            class_labels = outputs["instances"].pred_classes.to("cpu").numpy()
            
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)
            
            for i, prop in enumerate(props):
                object_number = i + 1
                area = prop.area
                centroid = prop.centroid
                bounding_box = prop.bbox
                
                class_name = 'Nuclei' if i < len(class_labels) else 'Unknown'
                
                csvwriter.writerow([image_filename, class_name, object_number, area, centroid, bounding_box])
    
    print("Object-level information saved to CSV file.")

def generate_plots(csv_path):
    df = pd.read_csv(csv_path)
    df['Base Name'] = df['File Name'].str.rsplit('_', 1).str[0]
    df['Category'] = df['File Name'].str.split('_').str[0]
    
    # Plot 1: Average number of Nuclei per image type
    avg_nuclei_per_base_name = df.groupby('Base Name')['Object Number'].mean().reset_index()
    plt.figure(figsize=(15, 6))
    sns.barplot(data=avg_nuclei_per_base_name, x='Base Name', y='Object Number')
    plt.title('Average Number of Nuclei per Image Type')
    plt.xticks(rotation=90)
    plt.show()
    
    # Plot 2: Average Nuclei area per image type
    avg_area_per_base_name = df.groupby('Base Name')['Area'].mean().reset_index()
    plt.figure(figsize=(15, 6))
    sns.barplot(data=avg_area_per_base_name, x='Base Name', y='Area')
    plt.title('Average Nuclei Area per Image Type')
    plt.xticks(rotation=90)
    plt.show()
    
    # Plot 3: Distribution of Number of Nuclei by Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Category', y='Object Number')
    plt.title('Distribution of Number of Nuclei by Category')
    plt.show()
    
    # Plot 4: Average area of Nuclei by Category
    avg_area_by_category = df.groupby('Category')['Area'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_area_by_category, x='Category', y='Area')
    plt.title('Average Area of Nuclei by Category')
    plt.show()

def save_labeled_masks(predictor, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_filename in os.listdir(input_dir):
        if not image_filename.lower().endswith('.png'):
            continue
        
        image_path = os.path.join(input_dir, image_filename)
        new_im = cv2.imread(image_path)
        outputs = predictor(new_im)
        
        binary_mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(np.uint8)
        combined_mask = np.sum(binary_mask, axis=0)
        labeled_mask = label(combined_mask)
        
        result_filename = os.path.splitext(image_filename)[0] + "_result.png"
        output_path = os.path.join(output_dir, result_filename)
        
        cv2.imwrite(output_path, labeled_mask.astype(np.uint16))
    
    print("Segmentation of all images completed.")

def main():
    cfg = setup_detectron()
    predictor = load_model(cfg)
    
    # Visualize random samples
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
    train_metadata = MetadataCatalog.get("my_dataset_train")
    visualize_random_samples(train_dataset_dicts, train_metadata)
    
    # Evaluate model
    evaluation_results = evaluate_model(cfg, predictor)
    print("Evaluation Results:", evaluation_results)
    
    # Process images and save object-level information
    process_images(predictor, TEST_IMAGES, OUTPUT_CSV)
    
    # Generate plots
    generate_plots(OUTPUT_CSV)
    
    # Save labeled masks
    save_labeled_masks(predictor, TEST_IMAGES, LABELED_MASKS_DIR)

if __name__ == "__main__":
    main()