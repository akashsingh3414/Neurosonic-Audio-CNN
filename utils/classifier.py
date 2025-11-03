import base64
import io
import numpy as np
import torch
import soundfile as sf
import librosa
from fastapi import HTTPException
from pathlib import Path
from utils.audio_processor import AudioProcessor
from utils.model import AudioCNN


class AudioClassifier:
    """Audio classifier for ESC-50 with preprocessing + inference."""

    def __init__(self, model_path=None):
        print("Initializing Audio Classifier...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if model_path is None:
            model_path = "./saved_models/best_model_20251103_185735_82.pth"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load model checkpoint and prepare processor."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classes = checkpoint.get("classes", [])
            if not self.classes:
                raise ValueError("No classes found in checkpoint!")

            sample_rate = checkpoint.get("sample_rate", 22050)
            mel_params = checkpoint.get("mel_params", {
                "n_fft": 2048, "hop_length": 512, "n_mels": 128, "f_min": 50, "f_max": sample_rate // 2
            })

            self.model = AudioCNN(num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device).eval()

            self.audio_processor = AudioProcessor(sample_rate=sample_rate, mel_params=mel_params)

            self.model_metadata = {
                "epoch": checkpoint.get("epoch", "unknown"),
                "val_accuracy": checkpoint.get("val_accuracy", "unknown"),
                "f1_score": checkpoint.get("f1_score", "unknown"),
                "sample_rate": sample_rate,
            }

            print(f"Loaded model with {len(self.classes)} classes")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, audio_b64: str):
        """Full inference pipeline"""
        try:
            audio_data, sample_rate = self._decode_audio(audio_b64)
            audio_data = self._to_mono(audio_data)
            audio_data, sample_rate = self._resample_audio(audio_data, sample_rate)
            chunks = self._chunk_audio(audio_data, sample_rate)

            all_probs, feature_maps, spec = self._run_inference(chunks)
            final_probs = self._aggregate_probs(all_probs)
            predictions = self._top_predictions(final_probs)
            viz_data = self._prepare_visualizations(feature_maps, spec, audio_data, sample_rate)

            return {"predictions": predictions, **viz_data}

        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    def _decode_audio(self, audio_b64: str):
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        print(f"Received audio: sr={sample_rate}, duration={len(audio_data)/sample_rate:.2f}s")
        return audio_data, sample_rate

    def _to_mono(self, audio_data):
        if audio_data.ndim > 1:
            print("Converted stereo to mono")
            audio_data = np.mean(audio_data, axis=1)
        return audio_data

    def _resample_audio(self, audio_data, sample_rate):
        target_sr = self.audio_processor.sample_rate
        if sample_rate != target_sr:
            print(f"Resampling: {sample_rate}Hz → {target_sr}Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        return audio_data, sample_rate

    def _chunk_audio(self, audio_data, sample_rate, chunk_seconds=5):
        target_len = int(chunk_seconds * sample_rate)
        total_len = len(audio_data)
        chunks = []

        if total_len > target_len:
            for i in range(total_len // target_len):
                chunks.append(audio_data[i * target_len:(i + 1) * target_len])
            rem = total_len % target_len
            if rem > sample_rate:
                pad_len = target_len - rem
                chunks.append(np.pad(audio_data[-rem:], (0, pad_len)))
        else:
            pad_len = target_len - total_len
            chunks = [np.pad(audio_data, (0, pad_len))]
        return chunks

    def _run_inference(self, chunks):
        all_probs, feature_maps = [], {}
        for chunk in chunks:
            spec = self.audio_processor.process_audio_chunk(chunk).to(self.device)
            with torch.no_grad():
                output, fmap = self.model(spec, return_feature_maps=True)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())
                feature_maps = fmap
        return np.vstack(all_probs), feature_maps, spec

    def _aggregate_probs(self, all_probs):
        confidences = np.max(all_probs, axis=1)
        weights = confidences / np.sum(confidences)
        final_probs = np.sum(all_probs * weights[:, None], axis=0)
        final_probs /= np.sum(final_probs)
        return final_probs

    def _top_predictions(self, final_probs, top_k=5):
        top_idx = np.argsort(final_probs)[::-1][:top_k]
        return [
            {"class": self.classes[i],
             "confidence": float(final_probs[i]),
             "confidence_pct": f"{final_probs[i] * 100:.2f}%"}
            for i in top_idx
        ]

    def _prepare_visualizations(self, feature_maps, spec, audio_data, sample_rate):
        viz_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:
                agg = torch.mean(tensor, dim=1)
                arr = np.nan_to_num(agg.squeeze(0).cpu().numpy())
                viz_data[name] = {"shape": list(arr.shape), "values": arr.tolist()}

        spec_np = spec.squeeze(0).squeeze(0).cpu().numpy()
        clean_spec = np.nan_to_num(spec_np)
        max_samples = 8000
        waveform_data = (
            audio_data[:: len(audio_data)//max_samples][:max_samples]
            if len(audio_data) > max_samples else audio_data
        )

        return {
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
            },
        }
