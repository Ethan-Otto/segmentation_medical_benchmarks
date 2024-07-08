import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import urllib.request
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def download_image(url, filename):
    urllib.request.urlretrieve(url, filename)

def main():
    # Download the image
    image_url = "http://images.cocodataset.org/val2017/000000439715.jpg"
    input_image = "input.jpg"
    download_image(image_url, input_image)

    print(os.getcwd())
    # Read the image
    im = cv2.imread(input_image)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Set up the detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Create the predictor
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    outputs = predictor(im)

    # Print the outputs
    print("Predicted classes:", outputs["instances"].pred_classes)
    print("Predicted boxes:", outputs["instances"].pred_boxes)

    # Visualize the results
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    print("Plotting Plot")
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.show()
    plt.savefig('./plots/detectron_test_result.png')

if __name__ == "__main__":
    main()