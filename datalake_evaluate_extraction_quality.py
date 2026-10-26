"""
datalake_evaluate_extraction_quality.py — Reviewer-requested item 4: extraction quality

Separates extraction error (OCR / Whisper) from downstream classification error, per
the reviewers' item 4. Reads two ground-truth manifests that must be
filled in by hand first (see the templates already created):

  evaluation/data_swamp_ocr_ground_truth.csv
      columns: file_name, field_name, ground_truth_value
      One row per rendered field per identity/financial document image. Fill in
      ground_truth_value by opening each image and reading off what is printed.

  evaluation/data_swamp_audio_ground_truth.csv
      columns: file_name, ground_truth_transcript
      One row per audio file. Fill in ground_truth_transcript by listening to each
      file and writing down exactly what is said.

Rows left blank, or marked with the MANUALLY_AUTHORED_PHASE1_FILE_... placeholder
(for the Phase 1 files whose original generation values were never captured by the
generator script), are skipped and counted separately rather than silently ignored.

This script does NOT re-run any of the four classification pipelines - it only reruns
the extraction step (Tesseract OCR / faster-whisper) fresh against data_swamp/, which
is cheap, so that OCR/WER can be scored against the ground truth independently of
whatever the pipelines' own metadata CSVs recorded.

Outputs (written to evaluation/output/):
  evaluation_ocr_field_accuracy_<ts>.csv   (file_name, field_name, ground_truth_value,
                                            ocr_text_excerpt, found_in_ocr_output,
                                            field_correct)
  evaluation_audio_wer_<ts>.csv            (file_name, ground_truth_transcript,
                                            whisper_transcript, wer)
  log_extraction_quality_<ts>.txt

Run:
  python datalake_evaluate_extraction_quality.py
"""

import io
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import jiwer

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR   = Path(__file__).parent
SWAMP_DIR  = BASE_DIR / "data_swamp"
EVAL_DIR   = BASE_DIR / "evaluation"
EVAL_OUT   = EVAL_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

OCR_MANIFEST   = EVAL_DIR / "data_swamp_ocr_ground_truth.csv"
AUDIO_MANIFEST = EVAL_DIR / "data_swamp_audio_ground_truth.csv"

PLACEHOLDER_PREFIX = "MANUALLY_AUTHORED_PHASE1_FILE"

WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False
    _WHISPER_MODEL = None


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("extraction_quality")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def is_fillable(value) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return False
    if s.startswith(PLACEHOLDER_PREFIX):
        return False
    return True


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def run_ocr_accuracy(logger: logging.Logger) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("OCR field-level accuracy")
    logger.info("=" * 70)

    if not OCR_MANIFEST.exists():
        logger.error(f"Manifest not found: {OCR_MANIFEST}")
        return pd.DataFrame()
    manifest = pd.read_csv(OCR_MANIFEST)

    fillable = manifest[manifest["ground_truth_value"].apply(is_fillable)]
    skipped = len(manifest) - len(fillable)
    logger.info(f"Manifest rows: {len(manifest)}  |  fillable: {len(fillable)}  |  skipped (blank/placeholder): {skipped}")
    if fillable.empty:
        logger.warning("No fillable ground-truth rows yet - nothing to score. Fill in "
                        f"{OCR_MANIFEST} and re-run.")
        return pd.DataFrame()

    ocr_cache: dict[str, str] = {}
    rows = []
    for _, row in fillable.iterrows():
        fn, field, gt = row["file_name"], row["field_name"], str(row["ground_truth_value"])
        if fn not in ocr_cache:
            path = SWAMP_DIR / fn
            if not OCR_AVAILABLE or not path.exists():
                ocr_cache[fn] = ""
                logger.warning(f"  Skipping OCR for {fn}: OCR unavailable or file missing")
            else:
                with Image.open(path) as img:
                    ocr_cache[fn] = pytesseract.image_to_string(img)
        ocr_text = ocr_cache[fn]
        found = normalise(gt) in normalise(ocr_text)
        rows.append({
            "file_name": fn, "field_name": field, "ground_truth_value": gt,
            "ocr_text_excerpt": ocr_text[:200].replace("\n", " "),
            "found_in_ocr_output": found, "field_correct": found,
        })
        logger.info(f"  {fn:<28} {field:<20} gt={gt!r:<30} found={found}")

    result_df = pd.DataFrame(rows)
    by_doc = result_df.groupby(result_df["file_name"])["field_correct"].mean()
    logger.info("\nField accuracy by document:")
    for fn, acc in by_doc.items():
        logger.info(f"  {fn:<28} {acc:.3f}")
    logger.info(f"\nOverall field accuracy: {result_df['field_correct'].mean():.3f} "
                f"({result_df['field_correct'].sum()}/{len(result_df)})")
    return result_df


def run_audio_wer(logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Audio word error rate (WER)")
    logger.info("=" * 70)

    if not AUDIO_MANIFEST.exists():
        logger.error(f"Manifest not found: {AUDIO_MANIFEST}")
        return pd.DataFrame()
    manifest = pd.read_csv(AUDIO_MANIFEST)

    fillable = manifest[manifest["ground_truth_transcript"].apply(is_fillable)]
    skipped = len(manifest) - len(fillable)
    logger.info(f"Manifest rows: {len(manifest)}  |  fillable: {len(fillable)}  |  skipped (blank/placeholder): {skipped}")
    if fillable.empty:
        logger.warning("No fillable ground-truth rows yet - nothing to score. Fill in "
                        f"{AUDIO_MANIFEST} and re-run.")
        return pd.DataFrame()

    rows = []
    for _, row in fillable.iterrows():
        fn, gt = row["file_name"], str(row["ground_truth_transcript"])
        path = SWAMP_DIR / fn
        if not WHISPER_AVAILABLE or not path.exists():
            logger.warning(f"  Skipping {fn}: Whisper unavailable or file missing")
            continue
        segments, _ = _WHISPER_MODEL.transcribe(str(path))
        hypothesis = " ".join(seg.text for seg in segments).strip()
        error = jiwer.wer(gt, hypothesis, reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM)
        rows.append({
            "file_name": fn, "ground_truth_transcript": gt,
            "whisper_transcript": hypothesis, "wer": round(error, 4),
        })
        logger.info(f"  {fn:<28} wer={error:.3f}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        logger.info(f"\nMean WER over {len(result_df)} scored files: {result_df['wer'].mean():.3f}")
    n_phase1_skipped = (manifest["file_name"].isin(
        ["Recording.wav", "call2.wav", "jane complaint.wav"]
    ) & ~manifest["ground_truth_transcript"].apply(is_fillable)).sum()
    if n_phase1_skipped:
        logger.info(
            f"NOTE: {n_phase1_skipped} Phase 1 (manually authored) audio file(s) have no "
            f"ground truth and were excluded, since their original transcripts were never "
            f"persisted at authoring time. Report WER for Phase 2 files only and state "
            f"this exclusion explicitly, per the brief."
        )
    return result_df


def main() -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    log_path = EVAL_OUT / f"log_extraction_quality_{TIMESTAMP}.txt"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("RO1 EXTRACTION QUALITY (reviewer item 4)")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"OCR available: {OCR_AVAILABLE}  |  Whisper available: {WHISPER_AVAILABLE}")
    logger.info("=" * 70)

    ocr_df = run_ocr_accuracy(logger)
    if not ocr_df.empty:
        ocr_df.to_csv(EVAL_OUT / f"evaluation_ocr_field_accuracy_{TIMESTAMP}.csv", index=False)

    wer_df = run_audio_wer(logger)
    if not wer_df.empty:
        wer_df.to_csv(EVAL_OUT / f"evaluation_audio_wer_{TIMESTAMP}.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
