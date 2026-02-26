# data/preprocess_streaming.py
"""
Streaming preprocessing for Kaggle/Colab — no full dataset download needed.
Processes IndicVoices-R Hindi directly from HuggingFace in streaming mode.

Usage (Kaggle/Colab):
    python data/preprocess_streaming.py \
        --output_dir /kaggle/working/processed \
        --max_samples 10000
"""

import argparse
import csv
import logging
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SR = 22050
TARGET_LUFS = -23.0
MIN_SNR = 20.0
MIN_DUR = 1.0
MAX_DUR = 15.0


def resample_and_normalise(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    meter = pyln.Meter(TARGET_SR)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness) and loudness > -70.0:
        audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio / peak * 0.99
    return audio.astype(np.float32)


def process_indicvoices_streaming(output_dir: Path, max_samples: int) -> None:
    wav_out = output_dir / "indicvoices" / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)
    meta_out = output_dir / "indicvoices" / "metadata.csv"

    logger.info("Loading IndicVoices-R in streaming mode...")
    dataset = load_dataset(
        "SPRINGLab/IndicVoices-R_Hindi",
        split="train",
        streaming=True,  # ← key: no full download
        trust_remote_code=True,
    )

    processed, skipped = 0, 0

    with open(meta_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["file_id", "text", "wav_path", "speaker_id", "gender"])

        for idx, sample in enumerate(
            tqdm(dataset, desc="IndicVoices-R", total=max_samples)
        ):
            if processed >= max_samples:
                break
            try:
                # Quality filters
                if sample.get("snr", 0) < MIN_SNR:
                    skipped += 1
                    continue
                dur = sample.get("duration", 0)
                if not (MIN_DUR <= dur <= MAX_DUR):
                    skipped += 1
                    continue

                text = sample.get("normalized") or sample.get("text", "")
                if not text.strip():
                    skipped += 1
                    continue

                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                audio = resample_and_normalise(audio, sr)
                file_id = f"indic_hi_{idx:06d}"
                out_path = wav_out / f"{file_id}.wav"
                sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

                writer.writerow(
                    [
                        file_id,
                        text,
                        str(out_path),
                        sample.get("speaker_id", ""),
                        sample.get("gender", ""),
                    ]
                )
                processed += 1

            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                skipped += 1

    logger.info(f"Done — processed: {processed}, skipped: {skipped}")


def process_ljspeech_streaming(output_dir: Path, max_samples: int) -> None:
    """Load LJSpeech directly from HuggingFace in streaming mode."""
    wav_out = output_dir / "ljspeech" / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)
    meta_out = output_dir / "ljspeech" / "metadata.csv"

    logger.info("Loading LJSpeech in streaming mode...")
    dataset = load_dataset(
        "keithito/lj_speech",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    processed, skipped = 0, 0

    with open(meta_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["file_id", "text", "wav_path"])

        for idx, sample in enumerate(tqdm(dataset, desc="LJSpeech", total=max_samples)):
            if processed >= max_samples:
                break
            try:
                text = sample.get("normalized_text") or sample.get("text", "")
                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                audio = resample_and_normalise(audio, sr)
                file_id = f"lj_{idx:05d}"
                out_path = wav_out / f"{file_id}.wav"
                sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

                writer.writerow([file_id, text, str(out_path)])
                processed += 1

            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                skipped += 1

    logger.info(f"Done — processed: {processed}, skipped: {skipped}")


def main(output_dir: str, max_samples: int) -> None:
    out = Path(output_dir)
    process_ljspeech_streaming(out, max_samples)
    process_indicvoices_streaming(out, max_samples)
    logger.info(f"All preprocessing complete → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Max samples per dataset. Start small (1000) to test.",
    )
    args = parser.parse_args()
    main(args.output_dir, args.max_samples)
