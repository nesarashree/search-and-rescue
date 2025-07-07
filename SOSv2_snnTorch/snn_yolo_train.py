'''
Spiking YOLOv3 Training Script (snn_yolo_train.py) - SOS.net (v2)
This script adapts the YOLOv3 object detection model for use with Spiking Neural Networks (SNNs), using snnTorch.
'''

import argparse
import torch
import torch.nn as nn
from snntorch import surrogate, functional as SF
from snntorch import Leaky # Explicitly import Leaky neuron
from models.common import Conv # Assumes 'models/common.py' contains the Conv module
from utils.datasets import create_dataloader # Assumes 'utils/datasets.py' for data loading
from utils.general import check_dataset, increment_path # Assumes 'utils/general.py' for utility functions
from utils.torch_utils import select_device # Assumes 'utils/torch_utils.py' for device selection
from train import train as yolo_train_base # Assumes 'train.py' as the base YOLOv3 training script

# SNN Specific Components

# Surrogate gradient for backpropagation through non-differentiable spiking events — fast_sigmoid.
spike_grad = surrogate.fast_sigmoid()

# Define a custom LIF neuron layer that integrates snnTorch's Leaky neuron.
class SNNLIFLayer(nn.Module):
    """
    Wrapper for snnTorch's Leaky Integrate-and-Fire (LIF) neuron.
    This layer will replace standard activation functions (e.g., ReLU) in the YOLOv3 model.
    """
    def __init__(self, beta=0.9, threshold=1.0, reset_mechanism="subtract"):
        """
        Initializes the SNNLIFLayer.
        args:
            beta (float): membrane potential decay rate (0 to 1). A higher beta means slower decay
            threshold (float): membrane potential threshold at which a spike is emitted
            reset_mechanism (str): mechanism to reset membrane potential after a spike ("subtract" or "zero")
        """
        super().__init__()
        # snn.Leaky initializes hidden states by default when init_hidden=True
        self.lif = Leaky(beta=beta, 
                         spike_grad=spike_grad, 
                         init_hidden=True, 
                         threshold=threshold,
                         reset_mechanism=reset_mechanism)
        # Store membrane and spike history if needed for analysis (not used in forward by default)
        self.mem = None
        self.spk = None

    def forward(self, x):
        """
        Forward pass for the LIF neuron layer.
        args:
            x (torch.Tensor): Input tensor (e.g., output from a convolutional layer)
        returns:
            torch.Tensor: Output spikes (binary tensor)
        """
        # The input 'x' here is the current injection to the LIF neuron.
        # The snn.Leaky module handles the integration and spiking logic.
        # It takes the input and the previous membrane potential (managed internally with init_hidden=True)
        # and returns the new spike and membrane potential.
        self.spk, self.mem = self.lif(x)
        return self.spk

# Replace ReLU with LIF in the standard YOLOv3 Conv layer.
class SNNConv(Conv):
    """
    Custom Convolutional layer for SNN-YOLOv3 that replaces the default activation (usually ReLU) with a Spiking Neural Network (SNN) LIF neuron.
    Inherits from the original YOLOv3 'Conv' module.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, beta=0.9, threshold=1.0, reset_mechanism="subtract"):
        """
        Initializes the SNNConv layer.
        args:
            c1 (int): input channels
            c2 (int): output channels
            k (int): kernel size
            s (int): stride
            p (int, optional): padding, defaults to None (calculated automatically)
            g (int): groups
            act (bool): whether to include an activation function — False here as it's replaced by SNNLIFLayer
            beta (float): membrane potential decay rate for the LIF neuron.
            threshold (float): threshold for the LIF neuron.
            reset_mechanism (str): reset mechanism for the LIF neuron.
        """
        # Call the parent Conv constructor, ensuring 'act' is False as we're replacing it.
        super().__init__(c1, c2, k, s, p, g, act=False) 
        # Instantiate the SNNLIFLayer to serve as the activation function.
        self.snn_act = SNNLIFLayer(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

    def forward(self, x):
        """
        Forward pass for the SNNConv layer.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output spikes from the LIF neuron.
        """
        # First, pass through the convolutional, batch normalization, and optional default activation (which is off).
        # The output of this super().forward(x) acts as the input current to the LIF neuron.
        return self.snn_act(super().forward(x))

# Recursively patch the YOLOv3 backbone to use SNNConv instead of regular Conv.
def convert_to_snn(model, beta=0.9, threshold=1.0, reset_mechanism="subtract"):
    """
    Recursively iterates through a PyTorch model and replaces all instances of the original 'Conv' layer with 'SNNConv' layers,
    injecting Leaky Integrate-and-Fire (LIF) neurons.
    args:
        model (torch.nn.Module): YOLOv3 PyTorch model to be converted.
        beta (float): membrane potential decay rate for the LIF neurons.
        threshold (float): threshold for the LIF neurons.
        reset_mechanism (str): reset mechanism for the LIF neurons.
    """
    for name, module in model.named_children():
        if isinstance(module, Conv):
            try:
                # assuming Conv stores its args like (c1, c2, k, s, p, g, act)
                args = module.conv.in_channels, module.conv.out_channels, \
                       module.conv.kernel_size[0], module.conv.stride[0], \
                       module.conv.padding[0], module.conv.groups, \
                       True # act is True for the original Conv, but we'll disable it in SNNConv
                
                # replace the original Conv with SNNConv
                setattr(model, name, SNNConv(*args[:6], act=False, # Pass original conv params, force act=False
                                             beta=beta, threshold=threshold, 
                                             reset_mechanism=reset_mechanism))
            except AttributeError:
                pass # placeholder

        elif len(list(module.children())) > 0: # check if module has children to recurse
            convert_to_snn(module, beta, threshold, reset_mechanism) # recurse into submodules

# Main training function for SNN-YOLOv3.
def train_snn_yolo(opt):
    """
    Args:
        opt (argparse.Namespace): Command-line arguments containing training configurations
    """
    # 1. Device Selection
    device = select_device(opt.device)
    print(f"[*] Using device: {device}")

    # 2. Load Pre-trained YOLOv3 Model
    # It's crucial that 'opt.weights' points to a standard YOLOv3 PyTorch checkpoint which contains a 'model' key. 
    # The .float().fuse() operations are common for optimizing YOLOv3 models.
    try:
        model = torch.load(opt.weights, map_location=device)['model'].float().fuse().to(device)
        print(f"[INFO] Loaded YOLOv3 model from {opt.weights}")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv3 model from {opt.weights}. Ensure it's a valid checkpoint.")
        print(f"Error: {e}")
        return

    # 3. Convert ANN YOLOv3 to SNN-YOLOv3
    # replaces all ReLU activations with LIF neurons.
    print("[INFO] Starting SNN conversion...")
    convert_to_snn(model, beta=0.9, threshold=1.0, reset_mechanism="subtract")
    print("[INFO] SNN conversion complete.")
    print(f"[INFO] Model architecture after SNN conversion:\n{model}") # Print for verification

    # 4. Prepare Save Directory
    # Increment path ensures that new training runs don't overwrite previous ones.
    opt.save_dir = str(increment_path(opt.project + '/' + opt.name, exist_ok=opt.exist_ok))
    print(f"[INFO] Training results will be saved to: {opt.save_dir}")

    # 5. Initiate Training
    # The 'yolo_train_base' function (imported from 'train.py') is expected to handle the main training loop, including data loading, optimization, loss calculation, and saving checkpoints.
    # It will now operate on the SNN-converted model.
    print("[INFO] Starting SNN-YOLOv3 training using the base YOLO training loop...")
    yolo_train_base(opt, model=model)
    print("[INFO] SNN-YOLOv3 training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Spiking YOLOv3 (SOS.net v2) model.")
    # --- General Training Arguments ---
    parser.add_argument('--weights', type=str, default='yolov3.pt', 
                        help='initial weights path for the base YOLOv3 model (ANN)')
    parser.add_argument('--cfg', type=str, default='models/yolov3.yaml', 
                        help='model.yaml path (defines YOLOv3 architecture)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', 
                        help='dataset.yaml path (defines dataset paths and classes)')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', 
                        help='hyperparameters path (e.g., learning rate, momentum)')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=416, 
                        help='train, val image size (pixels)')
    parser.add_argument('--project', default='runs/train', 
                        help='save results to project/name')
    parser.add_argument('--name', default='snn_yolov3', 
                        help='save results to project/name, e.g., runs/train/snn_yolov3')
    parser.add_argument('--device', default='', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', 
                        help='existing project/name ok, do not increment save_dir')
    
    # --- SNN Specific Arguments (Optional, but good for tuning) ---
    # These can be added if you want to expose LIF neuron parameters to the command line.
    # parser.add_argument('--snn-beta', type=float, default=0.9, 
    #                     help='membrane potential decay rate for LIF neurons (0 to 1)')
    # parser.add_argument('--snn-threshold', type=float, default=1.0, 
    #                     help='threshold for LIF neurons to emit a spike')
    # parser.add_argument('--snn-reset-mechanism', type=str, default='subtract', 
    #                     help='reset mechanism for LIF neurons ("subtract" or "zero")')

    opt = parser.parse_args()

    # If SNN-specific arguments were added, retrieve them here:
    # snn_beta = opt.snn_beta
    # snn_threshold = opt.snn_threshold
    # snn_reset_mechanism = opt.snn_reset_mechanism

    train_snn_yolo(opt)
    
'''
example usage:
python snn_yolo_train.py \
  --weights yolov3.pt \
  --data data/coco.yaml \
  --epochs 3 \
  --batch-size 16 \
  --imgsz 416 \
  --device 0
'''
