"""
DenseNet Inference API for KNative Deployment
Lightweight Flask application for serverless deployment
"""

import os
import time
import base64
import io
import json
import logging
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model_utils import load_optimized_model, get_imagenet_classes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
transform = None
classes = None
device = None

def initialize_model():
    """Initialize the DenseNet model and preprocessing"""
    global model, transform, classes, device
    
    logger.info("Initializing DenseNet model...")
    
    # Set device
    device = torch.device('cpu')  # Use CPU for lightweight deployment
    
    # Load model
    model = load_optimized_model(device)
    model.eval()
    
    # Set up preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet classes
    classes = get_imagenet_classes()
    
    logger.info("Model initialization complete")

def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess base64 encoded image"""
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        tensor = transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def run_inference(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Run inference on preprocessed image tensor"""
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor.to(device))
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
        
        predictions = []
        for i in range(5):
            class_idx = top5_indices[0][i].item()
            prob = top5_prob[0][i].item()
            class_name = classes.get(class_idx, f"class_{class_idx}")
            
            predictions.append({
                "class_id": class_idx,
                "class_name": class_name,
                "confidence": round(prob, 4)
            })
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "predictions": predictions,
        "latency_ms": latency_ms,
        "model_variant": "densenet121_optimized"
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not initialized"}), 503
        
        # Parse request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400
        
        image_data = data['image']
        batch_size = data.get('batch_size', 1)
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Run inference
        results = run_inference(image_tensor)
        
        # Add batch size info
        results['batch_size'] = batch_size
        
        return jsonify(results), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API info"""
    return jsonify({
        "service": "DenseNet Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Image classification (POST)",
            "/": "API information"
        },
        "model": "DenseNet-121 Optimized",
        "status": "ready" if model is not None else "initializing"
    })

if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)