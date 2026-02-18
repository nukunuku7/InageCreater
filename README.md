# 🎨 DEEPDANBOORU Image Creator Project

このプロジェクトは、

- DeepDanbooruタグ抽出  
- タグJSON生成  
- プロンプト自動生成  
- SDXL + ControlNet画像生成（GUI対応）

を統合した画像生成環境です。

---

# ✅ 必須環境

推奨環境：

- OS：Windows 10/11
- Python：3.11 以上
- GPU：NVIDIA推奨（VRAM 8GB以上）
- CUDA：11.8以上推奨

---

# 📂 フォルダ構造（変更禁止）

このプロジェクトは以下の構造を前提としています。  
**動作条件があります。**

```

DEEPDANBOORU/
│
├─ dataset/
│   ├─ images/              # タグ抽出対象画像
│   ├─ maked_tags/          # DeepDanbooruタグJSON出力
│   └─ prompts/             # 自動生成プロンプト保存先
│
├─ image_creater/
│   ├─ models/sdxl/
│   │   ├─ base/
│   │   │   └─ novaAnimeXL_ilv160.safetensors # SDXLベースモデル
│   │   │
│   │   └─ controlnet/      # ControlNetモデル群
│   │       ├─ Canny/
│   │       │   ├─ config.json
│   │       │   └─ diffusion_pytorch_model.safetensors # 輪郭から推論する学習済みモデル
│   │       │
│   │       └─ openpose/
│   │           ├─ config.json
│   │           └─ diffusion_pytorch_model.safetensors　# 骨格情報から推論する学習済みモデル 
│   │
│   ├─ outputs/             # 生成画像保存先
│   │
│   ├─ image_creater.py     # CLI実行用
│   └─ image_creater_GUI.py # GUI実行用
│
├─ tag_maker/
│   ├─ model/               # DeepDanbooruモデル
│   ├─ tags.txt             # プロンプトタグのすべて
│   └─ resnet.py
│
├─ project/
│   ├─ make_prompt.py       # プロンプト生成スクリプト
│   └─ make_tags_by_json.py # JSONタグ生成処理
│
├─ requirements.txt
└─ README.md

```

---

# ⬇️ 1. モデルのダウンロードと配置

---

## ✅ SDXLベースモデル

このプロジェクトでは以下を使用します：

- `novaAnimeXL_ilv160.safetensors`

配置場所：

```

image_creater/models/sdxl/base/novaAnimeXL_ilv160.safetensors

```

---

## ✅ ControlNetモデル

ControlNetを使う場合は以下を配置してください。

---

### Canny

配置：

```

image_creater/models/sdxl/controlnet/Canny/
├─ config.json
└─ diffusion_pytorch_model.safetensors

```

---

### OpenPose

配置：

```

image_creater/models/sdxl/controlnet/openpose/
├─ config.json
└─ diffusion_pytorch_model.safetensors

````

---

# ⚙️ 2. 環境構築

---

## 仮想環境作成（推奨）

```bash
python -m venv venv
````

有効化：

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 必要ライブラリのインストール

```bash
pip install -r requirements.txt
```

---

# 🚀 3. 実行手順

---

# ✅ Step 1 : タグ抽出（DeepDanbooru）

画像をここに入れます：

```
dataset/images/
```

タグ生成を実行：

```bash
python project/make_tags_by_json.py
```

出力：

```
dataset/maked_tags/*.json
```

---

# ✅ Step 2 : プロンプト生成

タグJSONからプロンプトを自動生成します。

```bash
python project/make_prompt.py
```

出力：

```
dataset/prompts/
```

---

# ✅ Step 3 : 画像生成（CLI）

```bash
python image_creater/image_creater.py
```

生成画像保存先：

```
image_creater/outputs/
```

---

# ✅ Step 4 : 画像生成（GUI）

GUI版を起動する場合：

```bash
python image_creater/image_creater_GUI.py
```

---

# ❗ よくあるエラー

---

## モデルが見つからない

```
FileNotFoundError: model not found
```

→ `image_creater/models/sdxl/base/` にモデルがあるか確認

---

## ControlNetが効かない

→ `config.json` と `diffusion_pytorch_model.safetensors` が揃っているか確認

---

## CUDAが使えない

```
Torch not compiled with CUDA enabled
```

→ GPU版PyTorchを入れてください

---

# 📌 今後追加予定

* LoRA対応
* 設定ファイル対応
