import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import LinkNeighborLoader

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import load_and_build_data

try:
    from models.txgnn_model import TxGNNModel
except ImportError:
    try:
        from models.txgnn import TxGNNModel
    except ImportError:
        print("❌ Error: Could not import TxGNNModel.")
        sys.exit(1)


def quick_eval(model, loader, device, target_rel_id, num_rels):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 1. 检查 edge_type 是否存在 (修复报错的关键)
            if not hasattr(batch, 'edge_type'):
                print("❌ Batch missing 'edge_type'. Check data construction.")
                return 0.0, 0.0

            # 2. 获取 Embedding
            # TxGNN 的 forward 需要 x, edge_index, edge_type
            if hasattr(model, 'encoder'):
                # 有些版本的 TxGNN 把 RGCN 放在 .encoder 下
                out = model.encoder(batch.x, batch.edge_index, batch.edge_type)
            else:
                out = model(batch.x, batch.edge_index, batch.edge_type)

            # 3. 构造边索引
            if hasattr(batch, 'edge_label_index'):
                src, dst = batch.edge_label_index
            else:
                src, dst = batch.src, batch.dst

            # 4. 【核心】强制使用目标 Relation ID
            batch_rel = torch.full((len(src),), target_rel_id, dtype=torch.long, device=device)

            # 5. 计算分数
            if hasattr(model, 'score'):
                scores = model.score(out, src, dst, batch_rel)
            else:
                scores = model.encoder.score(out, src, dst, batch_rel)

            prob = torch.sigmoid(scores)

            # 获取标签
            if hasattr(batch, 'edge_label'):
                lbl = batch.edge_label
            elif hasattr(batch, 'y'):
                lbl = batch.y
            else:
                lbl = batch.label

            preds.append(prob.cpu())
            targets.append(lbl.cpu())

    if not preds: return 0.5, 0.5

    all_preds = torch.cat(preds).numpy()
    all_targets = torch.cat(targets).numpy()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    return auc, 1 - auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 配置 ===
    hard_path = '../data/benchmark/Kaggle_drug_repositioning/test_hard.csv'
    model_path = '../model/TxGNN/txgnn_finetuned_best.pt'  # 确保模型路径正确

    # 基础数据路径
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'

    print(f"🚀 Scanning Relation IDs on HARD DATASET: {hard_path}")

    # 1. 加载全量图数据 (确保 data.edge_type 存在)
    # 我们把 hard_path 传给 test_path 参数，这样 datasets['test'] 就是我们要扫描的数据
    print("Loading data...", end='\r')
    data, datasets, num_nodes, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path=hard_path
    )
    print(f"Total unique relations: {num_rels}")

    # 2. 检查 Data 对象完整性
    if not hasattr(data, 'edge_type') or data.edge_type is None:
        print("❌ Critical Error: Loaded 'data' object is missing 'edge_type'!")
        return

    # 3. 初始化模型
    print("Loading Model...", end='\r')
    model = TxGNNModel(num_nodes, 128, num_rels, device=device).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        print(f"⚠️ Warning: Model weights not found at {model_path}. Using random weights (result will be random).")

    # 4. 创建 Loader (只取前 2000 条用于快速扫描)
    target_tensor = datasets['test']  # 这里装的是 Hard 数据
    # 截取前 2000 条
    limit = 2000
    if target_tensor[0].shape[1] > limit:
        print(f"Sampling first {limit} samples for speed...")
        subset_edge_label_index = target_tensor[0][:, :limit]
        subset_edge_label = target_tensor[1][:limit]
        subset_tensor = (subset_edge_label_index, subset_edge_label, target_tensor[2][:limit])
    else:
        subset_tensor = target_tensor

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[10, 5],
        edge_label_index=subset_tensor[0],
        edge_label=subset_tensor[1],
        batch_size=2048,
        shuffle=False
    )

    # 5. 开始扫描
    print(f"\nScanning IDs 0 to {num_rels - 1}...")
    print(f"{'ID':<5} | {'AUC':<10} | {'Inv-AUC':<10} | {'Note'}")
    print("-" * 45)

    results = []
    for r_id in range(num_rels):
        auc, inv_auc = quick_eval(model, loader, device, r_id, num_rels)

        note = ""
        if auc > 0.55: note = "Maybe"
        if auc > 0.65: note = "✅ MATCH!"
        if inv_auc > 0.65: note = "🔄 Inverted"

        print(f"{r_id:<5} | {auc:.4f}     | {inv_auc:.4f}     | {note}")
        results.append({'id': r_id, 'auc': auc})

    print("-" * 45)
    best = max(results, key=lambda x: x['auc'])
    print(f"\n🏆 Best Relation ID for Hard Dataset: {best['id']} (AUC={best['auc']:.4f})")

    if best['id'] == 7:
        print("✅ Confirmation: Hard dataset aligns with Cold Start dataset (ID 7).")
    else:
        print("⚠️ Surprise: Hard dataset uses a different relation ID!")


if __name__ == "__main__":
    main()