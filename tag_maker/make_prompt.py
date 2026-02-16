# make_prompt.py
import re
import json
from pathlib import Path

# ===== 設定 =====
TAGS_JSON_DIR = Path("C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/dataset/maked_tags")  # タグJSONフォルダ
PROMPTS_DIR = Path("C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/dataset/prompts")     # 出力プロンプトフォルダ

# SDXLのCLIP制限は77。SOS/EOSを除いた実質上限は75。
# 安全のため、1チャンクあたりのトークン上限を 65 に設定します。
MAX_CHUNK_TOKENS = 50
MAX_WEIGHTED_TAGS = 15 # 手描き感を出すため、強調できる枠を少し広げました
SORT_BY_SCORE = True
# =================

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== ① キャラ構造コア（最優先）=====
CHARACTER_CORE = [
    "(1girl:1.2)",
    "(highly detailed face:1.2)",
    "(perfect anatomy:1.2)",
    "(proper hands:1.1)",
    "(correct fingers:1.1)",
    "(detailed eyes:1.1)",
    "full body",
    "facing viewer"
]

# ===== ② スタイル・画質（手描き感を出すためのベース）=====
BASE_POSITIVE = [
    "masterpiece",
    "best quality",
    "extremely detailed",
    "(painterly style:1.2)",
    "(brush strokes:1.1)",
    "(canvas texture:1.0)",
    "(rich color:1.1)",
    "(atmospheric lighting:1.2)",
    "(dynamic lighting:1.1)",
    "sharp focus",
    "high resolution", # カンマを追加
    "detailed background"
]

# ===== ③ 最強のネガティブ（重複を削除し整理）=====
BASE_NEGATIVE = [
    "(worst quality, low quality:1.4)",
    "(deformed, distorted, disfigured:1.3)",
    "(bad anatomy, wrong anatomy:1.3)",
    "(mutated hands and fingers:1.4)",
    "(extra limbs, extra legs, extra arms:1.4)",
    "(missing limbs, amputee:1.3)",
    "(fused fingers, too many fingers:1.4)",
    "(unclear eyes:1.1)",
    "(plastic skin:1.2)",
    "(flat color:1.1)",
    "lowres", "blurry", "text", "signature", "watermark", "jpeg artifacts",
    "extra digits", "fewer digits", "bad hands", "malformed limbs", "sharp edge"
]

# ===== ユーティリティ =====
def clip_token_estimate(text: str) -> int:
    """
    記号を分離してトークン数をより正確に見積もる。
    CLIPトークナイザーの挙動に合わせ、カッコ・コロン・カンマをそれぞれ1と数える。
    """
    # 記号の周りにスペースを入れて分割しやすくする
    clean_text = re.sub(r"([(),:])", r" \1 ", text)
    tokens = clean_text.split()
    count = 0
    for t in tokens:
        # 長い単語や特殊な記号は分割されやすいため重めにカウント
        if len(t) > 8: count += 2
        else: count += 1
    return count

def normalize_tag(tag: str):
    exclude = {"solo", "rating:safe", "simple background", "looking at viewer"}
    if tag in exclude: return None
    return tag.replace("_", " ")

def format_tag(tag: str, score: float):
    if score >= 0.95: return f"({tag}:1.2)"
    elif score >= 0.9: return f"({tag}:1.1)"
    return tag

# ===== 汎用チャンク分割ロジック =====
def split_into_chunks(tag_list):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for tag in tag_list:
        # タグ自体のトークン + 区切りのカンマ(1)
        tokens = clip_token_estimate(tag) + 1
        
        if current_tokens + tokens > MAX_CHUNK_TOKENS:
            if current_chunk:
                chunks.append(", ".join(current_chunk))
            current_chunk = [tag]
            current_tokens = tokens
        else:
            current_chunk.append(tag)
            current_tokens += tokens
            
    if current_chunk:
        chunks.append(", ".join(current_chunk))
    return chunks

def build_prompts(data):
    # --- ポジティブの構築 ---
    pos_items = list(data.get("positive_tags", {}).items())
    if SORT_BY_SCORE:
        pos_items.sort(key=lambda x: x[1], reverse=True)

    full_positive_list = []
    full_positive_list.extend(CHARACTER_CORE)
    
    weighted_count = 0
    for tag, score in pos_items:
        tag = normalize_tag(tag)
        if not tag: continue
        
        # 指や腕の修正
        is_hand_related = any(x in tag for x in ["hand", "finger", "arm", "leg"])
        
        if is_hand_related:
            tag_text = f"({tag}:1.2)"
            is_weight = True
        else:
            is_weight = (weighted_count < MAX_WEIGHTED_TAGS and score >= 0.9)
            tag_text = format_tag(tag, score) if is_weight else tag
        
        full_positive_list.append(tag_text)
        if is_weight:
            weighted_count += 1
        
    full_positive_list.extend(BASE_POSITIVE)
    pos_chunks = split_into_chunks(full_positive_list)

    # --- ネガティブの構築 ---
    neg_chunks = split_into_chunks(BASE_NEGATIVE)

    return pos_chunks, neg_chunks

# ===== メイン処理 =====
for json_path in TAGS_JSON_DIR.glob("*.json"):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        
        # #### 修正：元のJSONから画像ファイル名を取得 ####
        original_image = data.get("image", "")

        positive_chunks, negative_chunks = build_prompts(data)

        # #### 修正：outputにimageキーを追加 ####
        output = {
            "image": original_image,
            "positive_chunks": positive_chunks,
            "negative_chunks": negative_chunks
        }

        out_path = PROMPTS_DIR / f"{json_path.stem}.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"[PROMPT] {out_path.name}")
        print(f"  + Pos: {len(positive_chunks)} chunks, - Neg: {len(negative_chunks)} chunks")
    except Exception as e:
        print(f"[ERROR] Failed to process {json_path.name}: {e}")

print("[DONE] 全てのプロンプト生成が完了しました。")