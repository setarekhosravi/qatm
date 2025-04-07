"""
    Created on Mon Apr 11:01:24 2025
    @author: STRH
    Single Object Tracking with QATM
"""

import cv2
import types
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from seaborn import color_palette

# Import functions from qatm_pytorch.py
from qatm_pytorch import CreateModel, ImageDataset, run_multi_sample, nms_multi, plot_result_multi

class QATMTracker:
    def __init__(self, video_path, use_cuda=True, alpha=25):
        self.video_path = video_path
        self.use_cuda = use_cuda
        self.alpha = alpha
        
        # Initialize model
        print("Initializing QATM model...")
        self.model = CreateModel(
            model=models.vgg19(pretrained=True).features, 
            alpha=self.alpha, 
            use_cuda=self.use_cuda
        )
        
        # Create result directory
        self.result_dir = 'tracking_results'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize template
        self.template = None
        self.template_path = os.path.join(self.result_dir, "template.jpg")
        
    def select_roi_and_save_template(self):
        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read the first frame")
            
        # Select ROI
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        
        # Extract template from ROI
        x, y, w, h = bbox
        self.template = frame[y:y+h, x:x+h]
        
        # Save template
        cv2.imwrite(self.template_path, self.template)
        print(f"Template saved to {self.template_path}")
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return bbox
        
    def track(self):
        # Select ROI in first frame
        initial_bbox = self.select_roi_and_save_template()
        
        # Create video writer for output
        output_path = os.path.join(self.result_dir, "tracked_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        
        # Process each frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}/{self.total_frames}")
            
            # Save current frame to temp file
            frame_path = os.path.join(self.result_dir, "current_frame.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Create dataset for current frame and template
            dataset = ImageDataset(Path(self.result_dir), frame_path, single_template="template.jpg")
            
            # Calculate score
            scores, w_array, h_array, thresh_list = run_multi_sample(self.model, dataset)
            
            # Apply NMS
            boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
            
            # Draw results on frame
            if len(boxes) > 0 and len(indices) > 0:
                result_frame = plot_result_multi(frame, boxes, indices, show=False)
                
                # Add frame number
                cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame to output video
                out.write(result_frame)
                
                # Display frame
                cv2.imshow("Tracking", result_frame)
            else:
                cv2.putText(frame, f"Frame: {frame_count} - Target lost", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)
                cv2.imshow("Tracking", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QATM Tracker')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--alpha', type=float, default=25, help='Alpha parameter for QATM')
    args = parser.parse_args()
    
    # Check CUDA availability
    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
    
    # Create and run tracker
    tracker = QATMTracker(args.video, use_cuda=use_cuda, alpha=args.alpha)
    tracker.track()
