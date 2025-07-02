# Spiking YOLOv3 Training Script (snn_yolo_train.py) - SOS.net (v2)
# This script adapts the YOLOv3 object detection model for use with Spiking Neural Networks (SNNs), using snnTorch.

import argparse
import torch
import torch.nn as nn
from snntorch import surrogate, functional as SF
from models.common import Conv
from utils.datasets import create_dataloader
from utils.general import check_dataset, increment_path
from utils.torch_utils import select_device
from train import train as yolo_train_base

import snntorch as snn

# Surrogate gradient
spike_grad = surrogate.fast_sigmoid()

# Define a LIF neuron layer
class SNNLIFLayer(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)

    def forward(self, x):
        spk, mem = self.lif(x)
        return spk

# Replace ReLU with LIF in Conv layer
class SNNConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=False)  # no ReLU
        self.snn_act = SNNLIFLayer()

    def forward(self, x):
        return self.snn_act(super().forward(x))

# Patch YOLOv3 backbone to use SNNConv instead of regular Conv recursively
def convert_to_snn(model):
    for name, module in model.named_children():
        if isinstance(module, Conv):
            args = module.args
            setattr(model, name, SNNConv(*args))
        else:
            convert_to_snn(module)

# Train SNN-YOLOv3
def train_snn_yolo(opt):
    device = select_device(opt.device)
    model = torch.load(opt.weights, map_location=device)['model'].float().fuse().to(device)
    
    convert_to_snn(model)
    
    print("[INFO] SNN conversion complete.")

    opt.name = increment_path(opt.project + '/' + opt.name, exist_ok=False)
    opt.save_dir = str(opt.name)

    # Train using standard training loop
    yolo_train_base(opt, model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov3.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=416, help='image size')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='snn_yolov3', help='experiment name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    train_snn_yolo(opt)
    
'''
python snn_yolo_train.py \
  --weights yolov3.pt \
  --data data/coco.yaml \
  --epochs 3 \
  --batch-size 16 \
  --imgsz 416 \
  --device 0
'''
