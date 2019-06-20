#!/usr/bin/env python3
# coding=utf-8
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument("--model_def", type=str, default="/home/nvidia/wali_ws/src/detect/config/yolov3.cfg",
                    help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="/home/nvidia/wali_ws/src/detect/weights/yolov3.weights",
                    help="path to weights file")
parser.add_argument("--class_path", type=str, default="/home/nvidia/wali_ws/src/detect/data/coco.names",
                    help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')

params = [
    # "--model_def", "/home/nvidia/wali_ws/src/detect/config/yolov3-tiny.cfg",
    # "--weights_path", "/home/nvidia/wali_ws/src/detect//weights/yolov3-tiny.weights",
    "--class_path", "/home/nvidia/wali_ws/src/detect/data/coco.names",
    "--batch_size", '1',
    '--cuda', '0',  # TX2 only has 1 GPU
]

args = parser.parse_args(params)
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up model
model = Darknet(args.model_def, img_size=args.img_size).to(device)

if args.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(args.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(args.weights_path))  # pytorch *.pth
print('load done!!!')

model.eval()  # Set in evaluation mode

classes = load_classes(args.class_path)  # Extracts class labels from file
# print(classes)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

out_path = '/home/nvidia/wali_ws/src/detect/scripts/output.png'


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)  # torch.nn.functional
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def yolov3_detect(img):
    """
    :param img: ori cv2.imread BGR order numpy
    :return: img with painted bboxes
    """
    # Configure input
    input_imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # H,W,C [0,255]
    input_imgs = transforms.ToTensor()(input_imgs)  # C,H,W [0.0, 1.0]
    # Pad to square resolution
    input_imgs, _ = pad_to_square(input_imgs, 0)
    # Resize
    input_imgs = resize(input_imgs, args.img_size)
    # add batch
    input_imgs = input_imgs.unsqueeze(0)
    # to cuda tensor
    input_imgs = Variable(input_imgs.type(Tensor))

    prev_time = time.time()  # previous
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, args.conf_thres, args.nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    print("\tInference Time: %s" % inference_time)

    detections = detections[0]

    res = cv2_detections(img, detections)

    return res


def cv2_detections(img, detections):
    """
    :param img: BGR order
    :param detections: bboxes of one img, detections[0]
    :return: img with painted bboxes
    """
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, args.img_size, img.shape[:2])  # input_size, ori_img_size
        unique_labels = detections[:, -1].cpu().unique()  # -1 is cls_pred
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)  # [R,G,B,1] [0.0,1.0]

        bbox_colors = [(int(255 * t[2]), int(255 * t[1]), int(255 * t[0])) for t in bbox_colors]  # tuple

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]  # index the color
            txt = classes[int(cls_pred)] + 'conf: %.3f' % float(cls_conf.cpu())
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=3)
            # cv2.line(img, pt1=(x1 + 5, y1 - 5), pt2=(102 + len(txt) * 30, y1 - 5), color=color, thickness=15)
            cv2.putText(img, text=txt, org=(x1, y1 - 3),  # FONT_HERSHEY_COMPLEX_SMALL
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255), thickness=1)

    return img


def plt_detections(img, detections):
    """
    :param img: RGB order
    :param detections: bboxes of one img, detections[0]
    :return:
    """
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)  # ori input cv img

    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, args.img_size, img.shape[:2])  # input_size, ori_img_size
        unique_labels = detections[:, -1].cpu().unique()  # -1 is cls_pred
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]  # index the color

            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1, y1,
                s=classes[int(cls_pred)] + 'conf: ' + cls_conf,  # add class name
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
