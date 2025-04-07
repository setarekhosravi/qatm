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
    cv2.imwrite("template/template.png", template)
    return template

def tracking(args):
    template_dir = "template/"
    result_path = args.result_images_dir
    if not os.path.isdir(result_path):
        os.mkdir(result_path)    

    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    
    print('One Sample Image Is Inputted')
    image_path = "sample/sample.png"
    dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
    print("calculate score...")
    scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
    print("nms...")
    boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
    d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
    del(dataset)
    return d_img, boxes #[x1, y1, x2, y2]

def yolo_detection(args, frame):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path= args.weights, force_reload= False)
    model.float()
    model.eval()    
    results = model(frame)
    bbox = results.xyxy[0].cpu().numpy()
    return bbox

def args_parser():
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--video_path', type=str, default='video.mp4')
    parser.add_argument('--weights', type= str, default='weights/yolov5s.pt')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-r', '--result_images_dir', default='result/')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        convert_video_to_frames(frame)
        if frame_number==0:
            bbox_det = yolo_detection(args, frame)
            x1,y1,x2,y2,_,_ = bbox_det[0]
        else:
            create_template(frame, [int(x1),int(y1),int(x2),int(y2)])
            d_img, bbox_tr = tracking(args)
            bbox_coords = bbox_tr[0]  # Get the first (and only) element
            x1, y1 = bbox_coords[0]   # First point (top-left)
            x2, y2 = bbox_coords[1]
        
        if frame_number == 0:
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.imshow('frame', annotated_frame)
        else:
            cv2.imshow('frame', d_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_number!=0:
            del(d_img)
            torch.cuda.empty_cache()
        frame_number +=1
    cap.release()
    cv2.destroyAllWindows()