from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.classifier import AudioClassifier


class InferenceRequest(BaseModel):
    audio_data: str


app = FastAPI(
    title="NeuroSonic Audio Classifier API",
    description="ESC-50 audio classification with CNN",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    classifier = AudioClassifier()
except Exception as e:
    print(f"ERROR: Failed to initialize classifier: {e}")
    classifier = None


@app.get("/")
def root():
    return {
        "name": "NeuroSonic Audio Classifier",
        "version": "2.0.0",
        "status": "running" if classifier else "model not loaded",
        "endpoints": {
            "health": "/health",
            "inference": "/inference (POST)",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
def health():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(classifier.device),
        "num_classes": len(classifier.classes)
    }


@app.get("/model-info")
def model_info():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "classes": classifier.classes,
        "metadata": classifier.model_metadata,
        "device": str(classifier.device),
        "sample_rate": classifier.audio_processor.sample_rate,
        "mel_params": classifier.audio_processor.mel_params
    }


@app.post("/inference")
def inference(request: InferenceRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.audio_data:
        raise HTTPException(status_code=400, detail="No audio data provided")
    return classifier.predict(request.audio_data)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting NeuroSonic Audio Classifier API")
    print("Docs → http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
