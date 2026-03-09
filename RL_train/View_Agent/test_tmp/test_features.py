"""
test_features.py - CLIP vs CLIP + DINOv2 特徵提取效果測試腳本

這個腳本可以用來測試單獨使用 CLIP，以及結合 CLIP + DINOv2 所萃取出的特徵對「遮擋」與「視角變化」的敏感度。

使用方式:
    1. 準備兩張圖片 (例如: img1.jpg 無遮擋, img2.jpg 有遮擋)
    2. 執行腳本:
       python test_features.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 將 3090server 加入路徑以可以 import FeatureExtractor
# 因為腳本放在 test_tmp 裡，所以往上一層找 3090server
server_path = Path(__file__).parent.parent / "3090server"
if str(server_path) not in sys.path:
    sys.path.insert(0, str(server_path))

from models.feature_extractor import FeatureExtractor


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """計算兩個特徵向量的餘弦相似度 (Cosine Similarity)"""
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def main():
    parser = argparse.ArgumentParser(description="Test CLIP and DINOv2 feature extraction")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image (e.g., clear view)")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image (e.g., occluded view)")
    args = parser.parse_args()

    # 1. 載入圖片
    try:
        img1 = Image.open(args.img1).convert("RGB")
        img2 = Image.open(args.img2).convert("RGB")
        print(f"✅ 成功載入圖片: \n  - {args.img1}\n  - {args.img2}")
    except Exception as e:
        print(f"❌ 圖片載入失敗: {e}")
        return

    # 2. 載入我們實作好的 FeatureExtractor (同時包含 CLIP 與 DINOv2)
    print("\n⏳ 正在載入模型 (CLIP-ViT-B/32 & DINOv2-base)...")
    t0 = time.time()
    extractor = FeatureExtractor()
    extractor._lazy_load() # 強制初始化
    print(f"✅ 模型載入完成 (耗時 {time.time()-t0:.2f} 秒, Device: {extractor._device})")

    # 3. 提取特徵 (使用 extractor 內部的私有方法分別取得 CLIP 和 DINOv2)
    print("\n⏳ 正在提取特徵...")
    
    # 圖 1 特徵
    clip_feat1 = extractor._clip_encode(img1)
    dino_feat1 = extractor._dino_encode(img1)
    fused_feat1 = extractor.extract(img1) # CLIP + DINOv2 concat
    
    # 圖 2 特徵
    clip_feat2 = extractor._clip_encode(img2)
    dino_feat2 = extractor._dino_encode(img2)
    fused_feat2 = extractor.extract(img2)

    # 4. 計算相似度
    # 相似度越高代表模型認為兩張圖片越「像」
    # 對於 View Agent 來說，我們希望模型能敏銳區分「無遮擋」跟「有遮擋」的差異（相似度降低）
    
    sim_clip = cosine_similarity(clip_feat1, clip_feat2)
    sim_dino = cosine_similarity(dino_feat1, dino_feat2)
    sim_fused = cosine_similarity(fused_feat1, fused_feat2)

    print("\n" + "="*50)
    print(" 📊 特徵比較結果 (Cosine Similarity)")
    print("="*50)
    print("相似度說明：數值越接近 1.0 代表兩張圖越相似。")
    print("若兩張圖是同一個視角但有/無遮檔，我們希望模型能察覺出差異。")
    print("-" * 50)
    
    print(f"🔹 CLIP 相似度 (語義):           {sim_clip:.4f}")
    print(f"🔹 DINOv2 相似度 (幾何/局部細節): {sim_dino:.4f}")
    print(f"🚀 Fused 相似度 (目前實際使用):   {sim_fused:.4f}")
    print("-" * 50)
    
    print("\n💡 分析建議:")
    if sim_clip > 0.95 and sim_dino < 0.90:
        print(" -> CLIP 覺得兩張圖很像 (可能都只看到『手臂』和『桌子』)，但 DINOv2 成功抓出了細節(遮擋)的差異。這正是我們選擇 Fused 的原因！")
    elif sim_dino > 0.95 and sim_clip < 0.90:
        print(" -> DINOv2 覺得空間結構沒變，但 CLIP 注意到了語義上的大幅改變。")
    else:
        print(" -> 這兩張圖片在 CLIP 和 DINOv2 眼中的差異幅度為一般情況。你可以換一組差異更大的照片試試。")


if __name__ == "__main__":
    main()
