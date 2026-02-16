# image_creater.py
import torch
import cv2
import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel
)

# ===== 設定 =====
BASE_MODEL_PATH = Path(
    "C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/image_creater/models/sdxl/base/novaAnimeXL_ilV160.safetensors"
) # SDXL Baseモデルパス

CONTROLNET_CANNY_PATH = Path(
    "C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/image_creater/models/sdxl/controlnet/Canny"
) # ControlNet Cannyモデルパス

IMAGE_DIR = Path(
    "C:/Users/nukunuku7/Pictures/iCloud Photos/Shared/2次元画像集"
) # 入力画像フォルダ

PROMPTS_DIR = Path(
    "C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/dataset/prompts"
) # プロンプトJSONフォルダ

OUTPUT_DIR = Path(
    "C:/Users/nukunuku7/.vscode/nukunuku/DeepDanbooru/image_creater/outputs"
) # 出力画像フォルダ

IMAGE_SIZE = 1024
STEPS = 30
CFG_SCALE = 7.0
CONTROL_STRENGTH = 0.7

BASE_SEED = 12345
NUM_RANDOM_PROMPTS = 1 # ランダムに選択するプロンプト数
IMAGES_PER_PROMPT = 3 # 1プロンプトあたりの生成画像数
# =================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== デバイス =====
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[INFO] Using device: {device}")

# ===== 1. ControlNet & Pipeline ロード =====
print(f"[INFO] Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_CANNY_PATH, 
    torch_dtype=dtype,
    use_safetensors=True
).to(device)

print(f"[INFO] Loading SDXL Pipeline...")
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    BASE_MODEL_PATH,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True
).to(device)

# ===== 2. アスペクト比を維持してサイズを計算する関数 (追加) =====
def get_scaled_dimensions(width, height, target_long_side=1024):
    """
    元画像の比率を維持しつつ、長辺を指定サイズに合わせて8の倍数で返す
    """
    aspect_ratio = width / height
    if width > height:
        target_w = target_long_side
        target_h = int(target_long_side / aspect_ratio)
    else:
        target_h = target_long_side
        target_w = int(target_long_side * aspect_ratio)
    
    # SDXLなどのモデルは8か64の倍数である必要があるため調整
    target_w = (target_w // 8) * 8
    target_h = (target_h // 8) * 8
    return target_w, target_h

# ===== 3. Cannyエッジ抽出関数 (修正) =====
def prepare_canny_condition(image_path: Path, target_w, target_h):
    """
    指定されたサイズに合わせてCannyエッジ画像を生成
    """
    if not image_path.exists():
        return None
    
    img = Image.open(image_path).convert("RGB")
    # 強制 1024x1024 ではなく、計算されたサイズにリサイズ
    img = img.resize((target_w, target_h), Image.LANCZOS)
    img_array = np.array(img)
    
    canny = cv2.Canny(img_array, 100, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    
    return Image.fromarray(canny)

# ===== 4. 長尺プロンプト対応関数 (以前のものを流用) =====
@torch.no_grad()
def encode_long_prompt(pos_chunks, neg_chunks):
    max_chunks = max(len(pos_chunks), len(neg_chunks))
    pos_chunks = pos_chunks + [""] * (max_chunks - len(pos_chunks))
    neg_chunks = neg_chunks + [""] * (max_chunks - len(neg_chunks))

    def get_combined_embeds(chunks, is_negative):
        token_embeds_list = []
        pooled_embeds_list = []
        for chunk in chunks:
            p_emb, n_emb, p_pool, n_pool = pipe.encode_prompt(
                prompt=chunk, negative_prompt="", device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=True
            )
            token_embeds_list.append(n_emb if is_negative else p_emb)
            pooled_embeds_list.append(n_pool if is_negative else p_pool)
        
        return torch.cat(token_embeds_list, dim=1), torch.mean(torch.stack(pooled_embeds_list), dim=0)

    p_emb, p_pool = get_combined_embeds(pos_chunks, False)
    n_emb, n_pool = get_combined_embeds(neg_chunks, True)
    return p_emb, p_pool, n_emb, n_pool

# ===== 5. メイン処理 =====
prompt_files = list(PROMPTS_DIR.glob("*.json"))
selected_prompts = random.sample(prompt_files, k=min(NUM_RANDOM_PROMPTS, len(prompt_files)))

global_index = 0

for prompt_path in selected_prompts:
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    original_img_name = data.get("image", "")
    print(f"\n[PROMPT] {prompt_path.stem}")

    canny_image = None
    # 生成サイズをデフォルトの正方形で初期化
    gen_w, gen_h = IMAGE_SIZE, IMAGE_SIZE 

    if original_img_name:
        original_img_path = IMAGE_DIR / original_img_name
        if not original_img_path.exists():
            stem = Path(original_img_name).stem
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".PNG", ".JPG"]:
                temp_path = IMAGE_DIR / f"{stem}{ext}"
                if temp_path.exists():
                    original_img_path = temp_path
                    break
        
        if original_img_path.exists():
            print(f"  [OK] Image found: {original_img_path.name}")
            # --- 修正箇所：元画像のサイズを取得して生成サイズを計算 ---
            with Image.open(original_img_path) as tmp_img:
                orig_w, orig_h = tmp_img.size
            gen_w, gen_h = get_scaled_dimensions(orig_w, orig_h, IMAGE_SIZE)
            print(f"  -> Setting generation size to: {gen_w}x{gen_h}")
            
            # 計算したサイズでCannyを作成
            canny_image = prepare_canny_condition(original_img_path, gen_w, gen_h)
        else:
            print(f"  [WARNING] File not found. Using default square size.")

    # 長尺プロンプト・エンコード (ここからインデントを調整)
    pos_chunks = data.get("positive_chunks", [])
    neg_chunks = data.get("negative_chunks", [])
    p_embeds, p_pool, n_embeds, n_pool = encode_long_prompt(pos_chunks, neg_chunks)

    # 画像生成ループ
    for i in range(IMAGES_PER_PROMPT):
        current_seed = BASE_SEED + global_index
        generator = torch.Generator(device).manual_seed(current_seed)

        pipe_kwargs = {
            "prompt_embeds": p_embeds,
            "pooled_prompt_embeds": p_pool,
            "negative_prompt_embeds": n_embeds,
            "negative_pooled_prompt_embeds": n_pool,
            "width": gen_w,   # 計算した幅を使用
            "height": gen_h,  # 計算した高さを使用
            "num_inference_steps": STEPS,
            "guidance_scale": CFG_SCALE,
            "generator": generator,
        }

        if canny_image is not None:
            pipe_kwargs["image"] = canny_image
            pipe_kwargs["controlnet_conditioning_scale"] = CONTROL_STRENGTH
        else:
            pipe_kwargs["image"] = Image.new("RGB", (gen_w, gen_h), (0, 0, 0))
            pipe_kwargs["controlnet_conditioning_scale"] = 0.0

        image = pipe(**pipe_kwargs).images[0]

        out_path = OUTPUT_DIR / f"{prompt_path.stem}_v{i+1:03d}.png"
        image.save(out_path)
        print(f"    Saved: {out_path.name} (Seed: {current_seed})")
        global_index += 1

print("[DONE] 全ての処理が完了しました。")
