import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from model import AudioCNN
from augmentation import NormalizeSpec


class AudioProcessor:
    """Audio preprocessing pipeline matching training configuration"""
    def __init__(self, sample_rate=22050, mel_params=None):
        if mel_params is None:
            mel_params = {
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "f_min": 50,  # Updated to match training
                "f_max": sample_rate // 2,
            }
        
        self.sample_rate = sample_rate
        self.mel_params = mel_params
        
        # Transform pipeline must exactly match validation transform in training
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=mel_params["n_fft"],
                hop_length=mel_params["hop_length"],
                n_mels=mel_params["n_mels"],
                f_min=mel_params["f_min"],
                f_max=mel_params["f_max"],
            ),
            T.AmplitudeToDB(),
            NormalizeSpec(),
        )

    def process_audio_chunk(self, audio_data):
        """Convert audio numpy array to spectrogram tensor"""
        # Ensure audio is 1D
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)  # Add batch dimension


class InferenceRequest(BaseModel):
    audio_data: str  # Base64 encoded audio file


class InferenceResponse(BaseModel):
    predictions: list
    visualization: dict
    input_spectrogram: dict
    waveform: dict
    metadata: dict


class AudioClassifier:
    """Audio classifier with improved error handling and model loading"""
    
    def __init__(self, model_path=None):
        print("Initializing Audio Classifier...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use specified model path or default
        if model_path is None:
            model_path = "./saved_models/best_model_20251027_182122_80_B.pth"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self._load_model(model_path)
        print("Model loaded successfully!")
    
    def _load_model(self, model_path):
        """Load model checkpoint and initialize processor"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            self.classes = checkpoint.get("classes", [])
            if not self.classes:
                raise ValueError("No classes found in checkpoint!")
            
            print(f"Loaded {len(self.classes)} classes")
            
            # Get audio processing parameters (use defaults if not in checkpoint)
            sample_rate = checkpoint.get("sample_rate", 22050)
            mel_params = checkpoint.get("mel_params", {
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "f_min": 50,
                "f_max": sample_rate // 2,
            })
            
            # Initialize model
            self.model = AudioCNN(num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize audio processor with same config as training
            self.audio_processor = AudioProcessor(
                sample_rate=sample_rate, 
                mel_params=mel_params
            )
            
            val_acc = checkpoint.get("val_accuracy", "unknown")
            if isinstance(val_acc, (float, int)):
                val_acc_display = int(np.ceil(val_acc))
            else:
                val_acc_display = val_acc

            self.model_metadata = {
                "epoch": checkpoint.get("epoch", "unknown"),
                "val_accuracy": val_acc_display,
                "f1_score": checkpoint.get("f1_score", "unknown"),
                "sample_rate": sample_rate,
            }
            
            print(f"Model metadata: {self.model_metadata}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, audio_b64: str):
        """
        Perform inference on base64-encoded audio
        
        Args:
            audio_b64: Base64 encoded audio file (wav, mp3, etc.)
            
        Returns:
            Dictionary with predictions and visualization data
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            
            print(f"Received audio: sr={sample_rate}, duration={len(audio_data)/sample_rate:.2f}s")

            # Convert stereo → mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("Converted stereo to mono")
            
            # Resample if necessary
            if sample_rate != self.audio_processor.sample_rate:
                print(f"Resampling: {sample_rate}Hz → {self.audio_processor.sample_rate}Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=self.audio_processor.sample_rate,
                )
                sample_rate = self.audio_processor.sample_rate

            # --- handle chunking and padding ---
            target_len = int(5 * sample_rate)
            total_len = len(audio_data)
            chunks = []

            if total_len > target_len:
                # Split into non-overlapping 5-second chunks
                num_chunks = total_len // target_len
                for i in range(num_chunks):
                    start = i * target_len
                    end = start + target_len
                    chunks.append(audio_data[start:end])
                # Add last partial chunk if > 1 sec
                rem = total_len % target_len
                if rem > sample_rate:  # >1s remainder
                    pad_len = target_len - rem
                    padded = np.pad(audio_data[-rem:], (0, pad_len))
                    chunks.append(padded)
            else:
                # Pad shorter audio to 5 seconds
                pad_len = target_len - total_len
                audio_data = np.pad(audio_data, (0, pad_len))
                chunks = [audio_data]

            all_probs = []
            feature_maps = {}

            for chunk in chunks:
                spec = self.audio_processor.process_audio_chunk(chunk).to(self.device)
                with torch.no_grad():
                    output, fmap = self.model(spec, return_feature_maps=True)
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    feature_maps = fmap  # keep last feature map for visualization

            probs = np.vstack(all_probs)  # (num_chunks, num_classes)

            confidences = np.max(probs, axis=1)
            weights = confidences / np.sum(confidences)
            final_probs = np.sum(probs * weights[:, None], axis=0)
            final_probs /= np.sum(final_probs)  # normalize

            TOP_CLASSES = 5
            top_idx = np.argsort(final_probs)[::-1][:TOP_CLASSES]

            predictions = [
                {
                    "class": self.classes[i],
                    "confidence": float(final_probs[i]),
                    "confidence_pct": f"{final_probs[i] * 100:.2f}%"
                }
                for i in top_idx
            ]
            
            # Prepare feature maps for visualization
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:  # [batch, channels, height, width]
                    # Average across channels for visualization
                    agg = torch.mean(tensor, dim=1)
                    arr = np.nan_to_num(agg.squeeze(0).cpu().numpy())
                    viz_data[name] = {"shape": list(arr.shape), "values": arr.tolist()}

            spec_np = spec.squeeze(0).squeeze(0).cpu().numpy()
            clean_spec = np.nan_to_num(spec_np)

            max_samples = 8000
            waveform_data = (
                audio_data[:: len(audio_data)//max_samples][:max_samples]
                if len(audio_data) > max_samples
                else audio_data
            )

            return {
                "predictions": predictions,
                "visualization": viz_data,
                "input_spectrogram": {
                    "shape": list(clean_spec.shape),
                    "values": clean_spec.tolist(),
                },
                "waveform": {
                    "values": waveform_data.tolist(),
                    "sample_rate": int(sample_rate),
                    "duration": float(len(audio_data) / sample_rate),
                },
                "metadata": {
                    "model_info": self.model_metadata,
                    "input_sample_rate": int(sample_rate),
                    "num_classes": len(self.classes),
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="NeuroSonic Audio Classifier API",
    description="ESC-50 audio classification with CNN",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier (will auto-load latest model)
try:
    classifier = AudioClassifier()
except Exception as e:
    print(f"ERROR: Failed to initialize classifier: {e}")
    classifier = None


@app.get("/")
def root():
    """Root endpoint with API information"""
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
    """Health check endpoint"""
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
    """Get detailed model information"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": classifier.classes,
        "metadata": classifier.model_metadata,
        "device": str(classifier.device),
        "sample_rate": classifier.audio_processor.sample_rate,
        "mel_params": classifier.audio_processor.mel_params
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    """
    Perform audio classification inference
    
    Args:
        request: JSON with base64-encoded audio_data
        
    Returns:
        Predictions with confidence scores and visualization data
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.audio_data:
        raise HTTPException(status_code=400, detail="No audio data provided")
    
    try:
        result = classifier.predict(request.audio_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("NeuroSonic Audio Classifier API Server")
    print("=" * 60)
    print("Starting server at http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )