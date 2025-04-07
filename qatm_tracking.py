"""
    Created on Mon Apr 11:01:24 2025
    @author: STRH
    Single Object Tracking with QATM
"""

import cv2
import types
import sys
import torch
from torchvision import models
import os
import gc
import glob
import argparse
from pathlib import Path
import ast
from alive_progress import alive_bar

# Import functions from qatm_pytorch.py using the same approach as in qatm.py
print("Importing qatm_pytorch.py...")
with open("qatm_pytorch.py") as f:
    p = ast.parse(f.read())

for node in p.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        p.body.remove(node)

module = types.ModuleType("mod")
code = compile(p, "mod.py", 'exec')
sys.modules["mod"] = module
exec(code, module.__dict__)

from mod import CreateModel, ImageDataset, run_multi_sample, nms_multi, plot_result_multi

def convert_video_to_frames(frame):
    """
    Convert a video to frames and save them in the specified output directory.
    """
    # Open the video file
    output_dir = "sample/"
    cv2.imwrite(output_dir + "sample.png", frame)


def create_template(frame, detection):
    """
    Create a template from a frame and a detection.
    """
    x1, y1, x2, y2 = detection
    template = frame[y1:y2, x1:x2]
    cv2.write("template/template.png", template)
    return template

def tracking(args):
    template_dir = "template/"
    result_path = args.result_images_dir
    if not os.path.isdir(result_path):
        os.mkdir(result_path)    

    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    
    print('One Sample Image Is Inputted')
    image_path = "sample/"
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
    print("calculate score...")
    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
    print("nms...")
    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
    d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
    return d_img, boxes #[x1, y1, x2, y2]

def args_parser():
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-r', '--result_images_dir', default='result/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    pass