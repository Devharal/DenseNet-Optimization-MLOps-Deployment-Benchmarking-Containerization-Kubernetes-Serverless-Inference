# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import base64
import io
from PIL import Image
import time
import logging
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DenseNet Inference API", version="1.0.0")

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image
    batch_size: int = 1

class PredictionResponse(BaseModel):
    predictions: list
    latency_ms: float
    model_variant: str
    confidence_scores: list

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.class_names = None
        self.model_variant = "densenet121_optimized"
        self._load_model()
        self._setup_transforms()
        self._load_imagenet_classes()

    def _load_model(self):
        """Load and optimize the DenseNet model"""
        logger.info(f"Loading model on device: {self.device}")
        
        # Load pre-trained DenseNet-121
        self.model = densenet121(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Apply optimizations
        self._optimize_model()
        
        logger.info("Model loaded and optimized successfully")

    def _optimize_model(self):
        """Apply various optimizations to the model"""
        # 1. TorchScript compilation
        self.model = torch.jit.script(self.model)
        
        # 2. Set to evaluation mode and disable gradients
        self.model.eval()
        
        # 3. Optimize for inference
        self.model = torch.jit.optimize_for_inference(self.model)
        
        logger.info("Model optimizations applied: TorchScript + optimize_for_inference")

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_imagenet_classes(self):
        """Load ImageNet class names"""
        try:
            # Try to load from file if available
            if os.path.exists('/app/imagenet_classes.json'):
                with open('/app/imagenet_classes.json', 'r') as f:
                    self.class_names = json.load(f)
            else:
                # Fallback to a few sample classes
                self.class_names = [f"class_{i}" for i in range(1000)]
        except Exception as e:
            logger.warning(f"Could not load class names: {e}")
            self.class_names = [f"class_{i}" for i in range(1000)]

    def _decode_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )

    def predict(self, base64_image: str, batch_size: int = 1) -> dict:
        """Make prediction on the input image"""
        start_time = time.time()
        
        try:
            # Decode and preprocess image
            image = self._decode_image(base64_image)
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Create batch if batch_size > 1
            if batch_size > 1:
                input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)
            
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions for each image in batch
                top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
                
                predictions = []
                confidence_scores = []
                
                for i in range(batch_size):
                    batch_predictions = []
                    batch_confidences = []
                    
                    for j in range(5):
                        class_idx = top5_indices[i][j].item()
                        confidence = top5_prob[i][j].item()
                        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
                        
                        batch_predictions.append({
                            "class": class_name,
                            "class_id": class_idx,
                            "confidence": round(confidence, 4)
                        })
                        batch_confidences.append(round(confidence, 4))
                    
                    predictions.append(batch_predictions)
                    confidence_scores.append(batch_confidences)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "predictions": predictions,
                "latency_ms": round(latency_ms, 2),
                "model_variant": self.model_variant,
                "confidence_scores": confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize model service
model_service = ModelService()

@app.get("/")
async def root():
    return {"message": "DenseNet Inference API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(model_service.device)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions on input image"""
    logger.info(f"Prediction request received for batch_size: {request.batch_size}")
    
    if request.batch_size < 1 or request.batch_size > 32:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 32")
    
    result = model_service.predict(request.image, request.batch_size)
    
    return PredictionResponse(**result)

@app.get("/model-info")
async def model_info():
    return {
        "model_variant": model_service.model_variant,
        "device": str(model_service.device),
        "num_classes": len(model_service.class_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)