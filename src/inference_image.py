#python inference_image.py --model ../outputs/best_model_iou.pth --input ../Custom_Valid_Images
import torch
import argparse
import cv2
import os

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir')
parser.add_argument(
    '--model',
    default='../outputs/model.pth',
    help='path to the model checkpoint'
)
args = parser.parse_args()

out_dir = os.path.join('..', 'CustomInferences_Result')
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    print(f"Image {i+1}")
    # Read the image.
    image = Image.open(os.path.join(args.input, image_path))

    image = image.resize((512, 512))

    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']
    segmented_image = draw_segmentation_map(outputs)
    
    final_image = image_overlay(image, segmented_image)
    # cv2.imshow('Segmented image', final_image)
    # cv2.waitKey(1)
    cv2.imwrite(os.path.join(out_dir, image_path), final_image)