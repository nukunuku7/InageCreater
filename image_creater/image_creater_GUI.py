import os
# --- 最優先: TensorFlowの初期化設定 (Ultra 7 向け設定) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# 初期化前に設定する必要がある項目
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except:
    pass

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import torch
import cv2
import numpy as np
import re
from pathlib import Path
from PIL import Image, ImageTk
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import threading
import gc

# ===== 設定 (パスなどはそのまま維持) =====
# SDXLベースモデル
BASE_MODEL_PATH = "C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/image_creater/models/sdxl/base/novaAnimeXL_ilV160.safetensors"

# ControlNetモデルのパス (CannyとOpenPose)
CN_MODELS = {
    "Canny (輪郭維持)": "C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/image_creater/models/sdxl/controlnet/Canny",
    "OpenPose (ポーズ維持)": "C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/image_creater/models/sdxl/controlnet/openpose"
}

# タグ推定モデルとタグリストのパス
TAGGER_MODEL_PATH = "C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/tag_maker/model/model-resnet_custom_v3.h5"
TAGS_LIST_PATH = "C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/tag_maker/model/tags.txt"

# 出力ディレクトリ
OUTPUT_DIR = Path("C:/Users/nukunuku7/.vscode/nukunuku/ImageCreater/image_creater/outputs")

# 生成設定
IMAGE_SIZE = 1024 # 画像の最大辺サイズ (512, 768, 1024など8の倍数で指定) - 大きすぎるとVRAM不足になる可能性があるため注意
MAX_CHUNK_TOKENS = 50 # プロンプトを分割する際の最大トークン数 (SDXLは約77トークンまでが安定しているため、余裕を持って50に設定)
STEPS = 30 # 推奨30～50 (RTX 5070の性能を活かすため、あまり多くしすぎない方が良い)
CFG_SCALE = 7.0 # 推奨6.0～8.0 (高すぎると不自然になる可能性があるため、7.0前後がおすすめ)


# 代表タグやプロンプトのコア部分 (これらは常に入れるようにする)
CHARACTER_CORE = ["(1girl:1.2)", "(highly detailed face:1.2)", "(perfect anatomy:1.2)", "(proper hands:1.1)", "full body", "facing viewer"]
BASE_POSITIVE = ["masterpiece", "best quality", "(painterly style:1.2)", "atmospheric lighting", "sharp focus"]
BASE_NEGATIVE = ["score_4, score_3, score_2", "(worst quality, low quality:1.4)", "(mutated hands and fingers:1.4)", "blurry", "text"]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nova Generator - Ultra 7 & RTX 5070 Optimized")
        self.geometry("950x950")
        self.configure(bg="#f5f5f5")
        
        # 状態管理
        self.tagger_model = None
        self.tags_list = []
        self.sd_pipe = None
        self.current_cn_mode = None
        self.selected_image_path = None
        self.scanning = False
        self.scan_line = None
        self.is_tagger_ready = False
        
        self.setup_ui()
        self.load_tagger_on_cpu()

    def setup_ui(self):
        # UI構築
        frame_top = tk.Frame(self, bg="#f5f5f5"); frame_top.pack(pady=10)
        self.btn_select = tk.Button(frame_top, text="参考画像を選択", command=self.select_image, width=20, state="disabled")
        self.btn_select.pack(side=tk.LEFT, padx=5)
        self.lbl_path = tk.Label(frame_top, text="モデル読み込み中...", fg="blue", bg="#f5f5f5")
        self.lbl_path.pack(side=tk.LEFT)

        frame_mode = tk.LabelFrame(self, text="生成設定", padx=10, pady=5, bg="#f5f5f5")
        frame_mode.pack(pady=5, fill=tk.X, padx=50)
        
        tk.Label(frame_mode, text="モード:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="Canny (輪郭維持)")
        self.combo_mode = ttk.Combobox(frame_mode, textvariable=self.mode_var, values=list(CN_MODELS.keys()), state="readonly", width=20)
        self.combo_mode.pack(side=tk.LEFT, padx=10)
        
        tk.Label(frame_mode, text="強度:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.scale_strength = tk.Scale(frame_mode, from_=0.0, to=1.5, resolution=0.1, orient=tk.HORIZONTAL, length=120, bg="#f5f5f5")
        self.scale_strength.set(0.7); self.scale_strength.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas.pack(pady=5)
        
        tk.Label(self, text="代表タグ:", font=("MS Gothic", 9), bg="#f5f5f5").pack()
        self.ent_char = tk.Entry(self, width=80, font=("Consolas", 12)); self.ent_char.pack(pady=5)
        
        tk.Label(self, text="ポジティブ:", bg="#f5f5f5").pack()
        self.txt_pos = scrolledtext.ScrolledText(self, height=10, width=110, font=("Consolas", 10)); self.txt_pos.pack(pady=5)
        
        tk.Label(self, text="ネガティブ:", bg="#f5f5f5").pack()
        self.txt_neg = scrolledtext.ScrolledText(self, height=5, width=110, font=("Consolas", 10)); self.txt_neg.pack(pady=5)

        self.btn_frame = tk.Frame(self, height=55); self.btn_frame.pack(pady=20, fill=tk.X, padx=50); self.btn_frame.pack_propagate(False)
        self.progress_canvas = tk.Canvas(self.btn_frame, bg="#95a5a6", height=55, highlightthickness=0)
        self.progress_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.progress_text = self.progress_canvas.create_text(425, 27, text="準備中...", fill="white", font=("MS Gothic", 12, "bold"))

        self.status = tk.Label(self, text=" System Booting...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_tagger_on_cpu(self):
        def _load():
            try:
                with open(TAGS_LIST_PATH, "r", encoding="utf-8") as f:
                    self.tags_list = [line.strip() for line in f]
                with tf.device('/CPU:0'):
                    self.tagger_model = tf.keras.models.load_model(TAGGER_MODEL_PATH, compile=False)
                self.is_tagger_ready = True
                self.after(0, self.on_tagger_ready)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load Error", f"Taggerロード失敗: {e}"))
        threading.Thread(target=_load, daemon=True).start()

    def on_tagger_ready(self):
        self.btn_select.config(state="normal")
        self.lbl_path.config(text="画像未選択", fg="gray")
        self.progress_canvas.config(bg="#3498db", cursor="hand2")
        self.progress_canvas.itemconfig(self.progress_text, text="✨ 画像を生成する")
        self.progress_canvas.bind("<Button-1>", lambda e: self.start_generation_thread())
        self.status.config(text=" Ready | Ultra 7 & RTX 5070 Mode")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.webp")])
        if not path: return
        self.selected_image_path = Path(path)
        self.lbl_path.config(text=self.selected_image_path.name, fg="black")
        
        # UI描画を止めないようにリサイズも別スレッドで
        def _preview():
            img = Image.open(path)
            img.thumbnail((300, 300))
            self.img_tk = ImageTk.PhotoImage(img)
            self.after(0, lambda: self._show_preview())
        threading.Thread(target=_preview, daemon=True).start()

    def _show_preview(self):
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.img_tk)
        self.start_scan_animation()
        threading.Thread(target=self.estimate_tags, daemon=True).start()

    def estimate_tags(self):
        if not self.tagger_model: return
        try:
            img = Image.open(self.selected_image_path).convert("RGB").resize((512, 512))
            img_input = np.asarray(img, dtype=np.float32) / 255.0
            with tf.device('/CPU:0'):
                preds = self.tagger_model(img_input[None, ...], training=False)[0].numpy()
            
            detected = [(self.tags_list[i].replace("_", " "), float(score)) for i, score in enumerate(preds) if score > 0.35]
            detected.sort(key=lambda x: x[1], reverse=True)
            
            char_tags = [t for t, s in detected[:10] if s > 0.6 and t not in ["1girl", "solo", "looking at viewer"]]
            
            self.after(0, lambda: self._fill_prompts(char_tags, detected))
        except Exception as e:
            print(f"Tagging Error: {e}")

    def _fill_prompts(self, char_tags, detected):
        self.ent_char.delete(0, tk.END)
        self.ent_char.insert(0, ", ".join(char_tags[:3]))
        
        pos_list = ["score_9, score_8_up, score_7_up, source_anime"] + CHARACTER_CORE
        pos_list += [f"({t}:1.1)" if s >= 0.9 else t for t, s in detected]
        pos_list.extend(BASE_POSITIVE)
        
        self.txt_pos.delete("1.0", tk.END); self.txt_pos.insert("1.0", ", ".join(pos_list))
        self.txt_neg.delete("1.0", tk.END); self.txt_neg.insert("1.0", ", ".join(BASE_NEGATIVE))
        self.scanning = False

    def run_generation(self):
        mode = self.mode_var.get()
        dtype = torch.float16

        # RTX 5070 最適化ロード
        if self.sd_pipe is None or self.current_cn_mode != mode:
            if self.sd_pipe: del self.sd_pipe; gc.collect(); torch.cuda.empty_cache()
            controlnet = ControlNetModel.from_pretrained(CN_MODELS[mode], torch_dtype=dtype).to("cuda")
            self.sd_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
            )
            self.sd_pipe.enable_model_cpu_offload() # 32GB RAMを活用してVRAM節約
            self.sd_pipe.enable_vae_tiling()
            self.current_cn_mode = mode

        raw_img = cv2.imdecode(np.fromfile(str(self.selected_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = raw_img.shape[:2]
        aspect = w / h
        gw, gh = (IMAGE_SIZE, int(IMAGE_SIZE/aspect)) if w > h else (int(IMAGE_SIZE*aspect), IMAGE_SIZE)
        gw, gh = (gw // 8) * 8, (gh // 8) * 8
        
        # Canny/Poseの前処理
        input_img = cv2.resize(raw_img, (gw, gh))
        if "Canny" in mode:
            cn_input = Image.fromarray(np.stack([cv2.Canny(input_img, 100, 200)]*3, axis=-1))
        else:
            cn_input = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        # プロンプト処理
        full_pos = f"({self.ent_char.get().strip()}:1.3), {self.txt_pos.get('1.0', tk.END).strip()}"
        pos_chunks = self.split_into_chunks(full_pos)
        neg_chunks = self.split_into_chunks(self.txt_neg.get('1.0', tk.END).strip())

        try:
            p_list, pp_list, n_list, np_list = [], [], [], []
            for pc, nc in zip(pos_chunks, neg_chunks):
                pe, ne, pp, np_e = self.sd_pipe.encode_prompt(pc, negative_prompt=nc, device="cuda")
                p_list.append(pe); pp_list.append(pp); n_list.append(ne); np_list.append(np_e)

            image = self.sd_pipe(
                prompt_embeds=torch.cat(p_list, dim=1), 
                pooled_prompt_embeds=torch.mean(torch.stack(pp_list), dim=0),
                negative_prompt_embeds=torch.cat(n_list, dim=1), 
                negative_pooled_prompt_embeds=torch.mean(torch.stack(np_list), dim=0),
                image=cn_input, width=gw, height=gh, num_inference_steps=STEPS, guidance_scale=CFG_SCALE,
                controlnet_conditioning_scale=self.scale_strength.get(),
                callback=lambda s, t, l: self.after(0, lambda: self.update_progress_ui(s / STEPS)), callback_steps=1
            ).images[0]

            out_name = f"nova_{self.selected_image_path.stem}.png"
            image.save(OUTPUT_DIR / out_name)
            torch.cuda.empty_cache(); gc.collect()
            self.after(0, lambda: self.finish(out_name))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Gen Error", str(e)))

    # --- アニメーションと補助 ---
    def start_scan_animation(self):
        self.scanning = True; self.scan_y, self.scan_dir = 0, 4; self.animate_scan()

    def animate_scan(self):
        if not self.scanning:
            if self.scan_line: self.canvas.delete(self.scan_line); self.scan_line = None
            return
        if self.scan_line: self.canvas.delete(self.scan_line)
        self.scan_y += self.scan_dir
        if self.scan_y >= 300 or self.scan_y <= 0: self.scan_dir *= -1
        self.scan_line = self.canvas.create_line(0, self.scan_y, 300, self.scan_y, fill="#00ffff", width=2)
        self.after(25, self.animate_scan)

    def start_generation_thread(self):
        if not self.selected_image_path or "Generating" in self.status.cget("text"): return
        self.status.config(text=" Generating on RTX 5070...")
        threading.Thread(target=self.run_generation, daemon=True).start()

    def update_progress_ui(self, ratio):
        w = self.progress_canvas.winfo_width()
        self.progress_canvas.delete("bar")
        self.progress_canvas.create_rectangle(0, 0, w * ratio, 55, fill="#27ae60", tags="bar")
        self.progress_canvas.itemconfig(self.progress_text, text=f"Generating... {int(ratio*100)}%")
        self.progress_canvas.tag_raise(self.progress_text)

    def finish(self, name):
        self.progress_canvas.delete("bar")
        self.progress_canvas.itemconfig(self.progress_text, text="✨ 画像を生成する")
        self.status.config(text=f" Saved: {name}")
        messagebox.showinfo("Success", f"生成完了: {name}")

    def split_into_chunks(self, full_text):
        tags = [t.strip() for t in full_text.replace("\n", ",").split(",") if t.strip()]
        chunks, cur, tokens = [], [], 0
        for tag in tags:
            t_count = len(re.sub(r"([(),:])", r" \1 ", tag).split()) + 1
            if tokens + t_count > MAX_CHUNK_TOKENS:
                chunks.append(", ".join(cur)); cur, tokens = [tag], t_count
            else: cur.append(tag); tokens += t_count
        if cur: chunks.append(", ".join(cur))
        return chunks

if __name__ == "__main__":
    App().mainloop()