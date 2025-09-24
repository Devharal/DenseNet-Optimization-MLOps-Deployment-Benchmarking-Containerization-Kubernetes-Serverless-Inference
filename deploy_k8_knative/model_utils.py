"""
Model utilities for DenseNet optimization and loading
"""

import torch
import torch.nn as nn
from torchvision import models
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def load_optimized_model(device: torch.device) -> nn.Module:
    """Load and optimize DenseNet-121 model"""
    logger.info("Loading DenseNet-121...")
    
    # Load pretrained model
    model = models.densenet121(pretrained=True)
    
    # Apply optimizations
    model = optimize_model(model)
    
    # Move to device
    model = model.to(device)
    
    return model

def optimize_model(model: nn.Module) -> nn.Module:
    """Apply lightweight optimizations to the model"""
    logger.info("Applying model optimizations...")
    
    # Set to eval mode
    model.eval()
    
    # Fuse conv-bn-relu layers for inference
    try:
        model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        logger.info("Applied TorchScript optimization")
    except Exception as e:
        logger.warning(f"TorchScript optimization failed: {e}")
        # Continue without TorchScript if it fails
    
    return model

def get_imagenet_classes() -> Dict[int, str]:
    """Get ImageNet class names"""
    # Simplified subset of ImageNet classes for demonstration
    # In production, you would load the full 1000 classes
    classes = {
        0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
        4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock",
        8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch",
        12: "house_finch", 13: "junco", 14: "indigo_bunting", 15: "robin",
        16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
        # Add more classes as needed
        281: "tabby_cat", 282: "tiger_cat", 283: "persian_cat", 284: "siamese_cat",
        285: "egyptian_cat", 286: "mountain_lion", 287: "lynx", 288: "leopard",
        # Common objects
        924: "guacamole", 925: "consomme", 926: "hot_pot", 927: "trifle",
        928: "ice_cream", 929: "ice_lolly", 930: "french_loaf", 931: "bagel",
    }
    
    # Fill remaining with generic names
    for i in range(1000):
        if i not in classes:
            classes[i] = f"class_{i}"
    
    return classes