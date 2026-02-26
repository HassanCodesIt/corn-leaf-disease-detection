import io
import os
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Allowed image types for upload
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"]

# Initialize FastAPI app
app = FastAPI(
    title="Corn Leaf Disease Detection API",
    description="Upload corn leaf images to detect diseases using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
MODEL_PATH = BASE_DIR / "corn_Leaf_model.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Class names for the corn leaf disease detection model
CLASS_NAMES = {
    0: "Brown Spot",
    1: "Corn Rust",
    2: "Corn Smut",
    3: "Downy Mildew",
    4: "Grey Leaf Spot",
    5: "Healthy",
    6: "Leaf Blight"
}


# Pydantic model for base64 image input
class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image data


# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def validate_image_type(content_type: str) -> None:
    """Validate that the uploaded file is an allowed image type."""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )


async def read_and_process_image(file: UploadFile) -> Image.Image:
    """Read uploaded file and convert to RGB PIL Image."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


@app.get("/")
async def root():
    """Redirect to the static frontend page."""
    from fastapi.responses import FileResponse
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict corn leaf disease from an uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with predictions and annotated image
    """
    validate_image_type(file.content_type)
    
    try:
        image = await read_and_process_image(file)
        
        # Run prediction
        results = model(image)
        
        # Extract predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                predictions.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, "Unknown"),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(bbox[0], 2),
                        "y1": round(bbox[1], 2),
                        "x2": round(bbox[2], 2),
                        "y2": round(bbox[3], 2)
                    }
                })
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predictions": predictions,
            "total_detections": len(predictions)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """
    Predict corn leaf disease and return the annotated image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        Annotated image with bounding boxes
    """
    validate_image_type(file.content_type)
    
    try:
        image = await read_and_process_image(file)
        
        # Run prediction
        results = model(image)
        
        # Get annotated image
        annotated_frame = results[0].plot()
        
        # Convert to PIL Image and then to bytes
        annotated_image = Image.fromarray(annotated_frame)
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=annotated_result.jpg"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def process_base64_image(base64_str: str) -> Image.Image:
    """Process base64 encoded image data and return PIL Image."""
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


@app.post("/predict/base64")
async def predict_base64(request: Base64ImageRequest):
    """
    Predict corn leaf disease from a base64 encoded image.
    Used for webcam frames and pasted images.
    
    Args:
        request: JSON body with base64 encoded image
        
    Returns:
        JSON response with predictions and base64 annotated image
    """
    try:
        image = process_base64_image(request.image)
        
        # Run prediction
        results = model(image)
        
        # Extract predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                predictions.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, "Unknown"),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(bbox[0], 2),
                        "y1": round(bbox[1], 2),
                        "x2": round(bbox[2], 2),
                        "y2": round(bbox[3], 2)
                    }
                })
        
        # Get annotated image
        annotated_frame = results[0].plot()
        
        # Convert to PIL Image and then to base64
        annotated_image = Image.fromarray(annotated_frame)
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)
        annotated_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        
        return JSONResponse(content={
            "success": True,
            "predictions": predictions,
            "total_detections": len(predictions),
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
