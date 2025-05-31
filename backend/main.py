from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import uvicorn

app = FastAPI(title="Satellite Image Classifier API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = "tuned_deep_seq_model.h5"
model = None

CLASS_NAMES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = np.array(image)
        
        # Resize to model input size (64x64)
        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Satellite Image Classifier API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded"
    }

@app.get("/classes")
async def get_classes():
    """Get available classification classes"""
    return {"classes": CLASS_NAMES}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict land use class from uploaded satellite image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        probabilities = predictions[0]
        
        # Format results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                "class": class_name,
                "confidence": float(probabilities[i]),
                "percentage": f"{probabilities[i] * 100:.2f}%"
            })
        
        # Sort by confidence
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # Get top prediction
        top_prediction = results[0]
        
        return {
            "success": True,
            "filename": file.filename,
            "top_prediction": top_prediction,
            "all_predictions": results,
            "model_info": {
                "input_size": "64x64",
                "classes": len(CLASS_NAMES)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Predict multiple images at once"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            predictions = model.predict(processed_image)
            probabilities = predictions[0]
            
            top_class_idx = np.argmax(probabilities)
            
            results.append({
                "filename": file.filename,
                "predicted_class": CLASS_NAMES[top_class_idx],
                "confidence": float(probabilities[top_class_idx]),
                "all_probabilities": {
                    CLASS_NAMES[i]: float(probabilities[i]) 
                    for i in range(len(CLASS_NAMES))
                }
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)