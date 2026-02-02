"""recommend.py
Recommend top-k eyeglass frames for a given face photo using Google Gemini's
vision API instead of a local trained regressor.

Requirements:
    pip install google-genai pandas Pillow

Usage:
    Set your API key:
        export GEMINI_API_KEY="your-key-here"

    Run:
        python recommend.py --face-image path/to/face.jpg --top-k 5

    Optionally point to a custom frame catalog:
        python recommend.py --face-image path/to/face.jpg --catalog data/frame_catalog.csv --top-k 3
"""

import json
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-2.0-flash"          # adjust to whichever model you want
DEFAULT_CATALOG = Path(__file__).parent.parent / "data" / "frame_catalog.csv"

# Columns from frame_catalog.csv that we pass to Gemini as context.
# Adjust this list to match whatever columns your catalog actually has.
FRAME_CONTEXT_COLS = [
    "FrameID", "Brand", "Shape", "Color", "Material",
    "RimStyle", "BridgeType",
    "Width_mm", "LensHeight_mm", "LensWidth_mm",
    "NoseBridgeWidth_mm", "TempleLength_mm",
]


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def _build_client() -> genai.Client:
    """Create a Gemini client.  API key is read from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY environment variable is not set.\n"
            "  export GEMINI_API_KEY='your-key-here'"
        )
    return genai.Client(api_key=api_key)


def _read_face_image(path: Path) -> bytes:
    """Read a face image from disk and return raw bytes."""
    if not path.exists():
        sys.exit(f"ERROR: Face image not found: {path}")
    return path.read_bytes()


def _mime_for(path: Path) -> str:
    """Guess MIME type from file extension."""
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    return mapping.get(path.suffix.lower(), "image/jpeg")


def score_frames(
    client: genai.Client,
    face_image_bytes: bytes,
    face_mime: str,
    frames_df: pd.DataFrame,
) -> pd.Series:
    """Send the face image + all frame specs to Gemini in ONE request and get
    back a numeric compatibility score (0–100) for every frame.

    Returns a pandas Series indexed the same as frames_df.
    """

    # --- build a compact text table of frame specs ---
    cols = [c for c in FRAME_CONTEXT_COLS if c in frames_df.columns]
    frame_table = frames_df[cols].to_string(index=False)

    prompt = (
        "You are an expert optician and eyeglass stylist.\n\n"
        "Below is a photo of a person's face, followed by a catalog of eyeglass frames "
        "with their measurements and properties.\n\n"
        "For EACH frame in the catalog, output a single compatibility score from 0 to 100 "
        "indicating how well that frame would suit this face. Consider face shape, "
        "proportions, symmetry, style, and the frame's dimensions and design.\n\n"
        "IMPORTANT – respond with ONLY a valid JSON array of objects, one per frame, "
        "in the SAME ORDER as the catalog. Each object must have exactly two keys:\n"
        '  "FrameID" : the frame\'s ID (string or number, copied exactly from the table)\n'
        '  "score"   : an integer 0-100\n\n'
        "Do NOT include any other text, markdown fencing, or explanation.\n\n"
        "--- FRAME CATALOG ---\n"
        f"{frame_table}\n"
        "--- END CATALOG ---\n"
    )

    # --- call Gemini with inline image ---
    contents = [
        types.Part.from_bytes(data=face_image_bytes, mime_type=face_mime),
        prompt,
    ]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",  # ask for structured JSON back
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    # --- parse the JSON array ---
    raw = response.text.strip()
    # strip optional markdown fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    scores_list = json.loads(raw)

    # Build a mapping FrameID -> score
    score_map = {str(item["FrameID"]): int(item["score"]) for item in scores_list}

    # Map back onto the DataFrame (default 0 if Gemini missed a frame)
    return frames_df["FrameID"].astype(str).map(score_map).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Main recommendation logic
# ---------------------------------------------------------------------------
def recommend(
    face_image_path: Path,
    catalog_path: Path = DEFAULT_CATALOG,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return the top-k recommended frames as a DataFrame."""

    # 1. Load catalog
    if not catalog_path.exists():
        sys.exit(f"ERROR: Frame catalog not found: {catalog_path}")
    frames_df = pd.read_csv(catalog_path)
    print(f"[INFO] Loaded {len(frames_df)} frames from {catalog_path}")

    # 2. Read face image
    face_bytes = _read_face_image(face_image_path)
    face_mime = _mime_for(face_image_path)
    print(f"[INFO] Loaded face image ({len(face_bytes):,} bytes, {face_mime})")

    # 3. Score every frame via Gemini
    client = _build_client()
    print(f"[INFO] Sending request to Gemini ({MODEL})…")
    scores = score_frames(client, face_bytes, face_mime, frames_df)
    frames_df["PredictedBeautyScore"] = scores

    # 4. Sort and return top-k
    top = (
        frames_df.sort_values("PredictedBeautyScore", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # Keep a readable subset of columns for display
    display_cols = ["FrameID"] + [
        c for c in ["Brand", "Shape", "Color"] if c in top.columns
    ] + ["PredictedBeautyScore"]

    print(f"\n[DEBUG] Score range: min={scores.min()}, max={scores.max()}, "
          f"std={scores.std():.2f}")
    print(f"[DEBUG] Top {top_k} frame IDs: {top['FrameID'].tolist()}\n")

    return top[display_cols]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(
        description="Recommend eyeglass frames for a face using Gemini vision."
    )
    parser.add_argument(
        "--face-image", type=str, required=True,
        help="Path to a photo of the person's face (jpg/png/webp).",
    )
    parser.add_argument(
        "--catalog", type=str, default=str(DEFAULT_CATALOG),
        help="Path to frame_catalog.csv (default: data/frame_catalog.csv).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top frames to return (default: 5).",
    )
    args = parser.parse_args()

    top = recommend(
        face_image_path=Path(args.face_image),
        catalog_path=Path(args.catalog),
        top_k=args.top_k,
    )
    print(top.to_string(index=False))


if __name__ == "__main__":
    cli()