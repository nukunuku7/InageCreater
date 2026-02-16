# make_tags_by_json.py
import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ===== 設定 =====
IMAGE_DIR = Path("C:/Users/nukunuku7/Pictures/iCloud Photos/Shared/2次元画像集") # 入力画像フォルダ
MODEL_PATH = Path("C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/tag_maker/model/model-resnet_custom_v3.h5") # 学習済みモデルのパス
TAGS_PATH = Path("C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/tag_maker/model/tags.txt") # タグ一覧のパス
OUTPUT_DIR = Path("C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/dataset/maked_tags") # 出力フォルダ

IMAGE_SIZE = 512
THRESHOLD_POSITIVE = 0.5
THRESHOLD_NEGATIVE = 0.3
# 光・影・タッチなどのニュアンスを拾うための低めの閾値
THRESHOLD_NUANCE = 0.35 

# 抽出したいキーワード
NUANCE_KEYWORDS = [
    "lighting", "shadow", "sunlight", "glow", "atmosphere", # 光・影
    "style", "sketch", "watercolor", "oil", "art", "shading", "lineart" # タッチ
]
# =================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU設定（変更なし）
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# タグ一覧ロード
with open(TAGS_PATH, "r", encoding="utf-8") as f:
    TAGS = [line.strip() for line in f]

# モデルロード
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.trainable = False

# 画像一覧
image_files = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png", ".webp")])

def preprocess_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img[None, ...]

print(f"[INFO] Processing {len(image_files)} images...")

for idx, path in enumerate(image_files, 1):
    img = preprocess_image(path)
    preds = model(img, training=False)[0].numpy()

    positive = {}
    negative = {}
    nuance_tags = {} # 光・影・タッチ専用

    for tag, score in zip(TAGS, preds):
        score = float(score)
        
        # キーワードに合致するかチェック
        is_nuance = any(k in tag for k in NUANCE_KEYWORDS)

        if score >= THRESHOLD_POSITIVE:
            positive[tag] = score
        elif is_nuance and score >= THRESHOLD_NUANCE:
            # 光やタッチに関するタグなら、少し低いスコアでも拾う
            nuance_tags[tag] = score
        elif score >= THRESHOLD_NEGATIVE:
            negative[tag] = score

    # nuance_tags を positive に統合（重複しないように）
    positive.update(nuance_tags)

    out_path = OUTPUT_DIR / f"{path.stem}.json"
    out_path.write_text(
        json.dumps({
            "image": path.name,
            "positive_tags": positive,
            "negative_tags": negative,
            "detected_nuance_count": len(nuance_tags) # デバッグ用
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if idx % 10 == 0 or idx == len(image_files):
        print(f"[{idx}/{len(image_files)}] TAGGED: {path.name} (Nuances: {len(nuance_tags)})")

print("[DONE]")
        