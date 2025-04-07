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
import ast

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

