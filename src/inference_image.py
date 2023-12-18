#python inference_image.py --model ../outputs/best_model_iou.pth --input ../Custom_Valid_Images
from matplotlib import pyplot as plt
import torch
import argparse
import cv2
import os

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model
def image_segmentation(image_path):
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input dir')
    parser.add_argument(
        '--model',
        default='../outputs/model.pth',
        help='path to the model checkpoint'
    )
    args = parser.parse_args()
    # Set computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
    ckpt = torch.load('../../outputs/best_model_iou.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    # Read the image.
    image = Image.open(image_path)

    image = image.resize((512, 512))

    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']
    segmented_image = draw_segmentation_map(outputs)
       
    final_image = image_overlay(image, segmented_image)
    # Display the image using matplotlib
    return final_image
