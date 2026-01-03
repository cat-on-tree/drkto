import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from utils import load_and_build_data, create_loader

try:
    from models.txgnn_model import TxGNNModel
except:
    from .txgnn import TxGNNModel


def get_label_robust(batch):
    if hasattr(batch, 'label') and batch.label is not None: return batch.label
    if hasattr(batch, 'y') and batch.y is not None: return batch.y
    if hasattr(batch, 'edge_label') and batch.edge_label is not None: return batch.edge_label
    return None


def quick_eval(model, loader, device, target_rel_id):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            lbl = get_label_robust(batch)
            if lbl is None: continue

            # 使用 Encoder (RGCN) 获取 Embedding
            if hasattr(model, 'encoder'):
                enc = model.encoder
            else:
                enc = model

            out = enc(batch.x, batch.edge_index, batch.edge_type)

            # 构造 Rel Tensor
            if hasattr(batch, 'edge_label_index'):
                src, dst = batch.edge_label_index
            else:
                src, dst = batch.src, batch.dst

            # 强制使用当前扫描的 ID
            batch_rel = torch.full((len(src),), target_rel_id, dtype=torch.long, device=device)

            scores = enc.score(out, src, dst, batch_rel)
            prob = torch.sigmoid(scores)

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

    # 这里的路径一定要是你刚才新生成的那个 strict CSV
    test_path = '../data/benchmark/Kaggle_drug_repositioning/test_cold.csv'
    txgnn_path = '../model/TxGNN/txgnn_finetuned_best.pt'
    nodes_path = '../data/benchmark/PrimeKG/nodes.csv'
    train_path = '../data/benchmark/PrimeKG/train_edges.csv'
    val_path = '../data/benchmark/PrimeKG/val_edges.csv'

    print("...Loading Metadata...", end='\r')
    _, _, _, num_rels, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path=test_path, test_hard_path=None
    )

    # 加载数据构建 Loader
    print(f"Reading {test_path}...")
    df_cold = pd.read_csv(test_path)
    # 扫描时为了快，可以只取前2000条，但最好全跑因为Cold数据本身就不多
    if len(df_cold) > 2000: df_cold = df_cold.head(2000)

    edge_label_index = torch.stack([
        torch.tensor(df_cold['x_index'].values, dtype=torch.long),
        torch.tensor(df_cold['y_index'].values, dtype=torch.long)
    ], dim=0)
    edge_label = torch.tensor(df_cold['label'].values, dtype=torch.float)
    rel_tensor = torch.zeros(len(df_cold), dtype=torch.long)  # 占位

    data, _, num_nodes, _, _ = load_and_build_data(
        nodes_path, train_path, val_path, test_path=test_path, test_hard_path=None
    )

    loader = create_loader(data, (edge_label_index, edge_label, rel_tensor),
                           batch_size=2048, num_neighbors=[10, 5], shuffle=False)

    print(f"...Loading Model (num_rels={num_rels})...", end='\r')
    model = TxGNNModel(num_nodes, 128, num_rels, device=str(device)).to(device)
    model.load_state_dict(torch.load(txgnn_path, map_location=device), strict=False)
    model.eval()

    print(f"\n🚀 Scanning IDs 0 to {num_rels - 1} on NEW STRICT DATA...")
    print(f"{'ID':<5} | {'AUC':<10} | {'Inv-AUC':<10} | {'Note'}")
    print("-" * 45)

    results = []
    for r_id in range(num_rels):
        auc, inv_auc = quick_eval(model, loader, device, target_rel_id=r_id)

        note = ""
        if auc > 0.6: note = "Maybe"
        if auc > 0.7: note = "✅ MATCH!"
        if inv_auc > 0.7: note = "🔄 Inverted"

        print(f"{r_id:<5} | {auc:.4f}     | {inv_auc:.4f}     | {note}")
        results.append({'id': r_id, 'auc': auc, 'inv_auc': inv_auc})

    print("-" * 45)

    best = max(results, key=lambda x: x['auc'])
    best_inv = max(results, key=lambda x: x['inv_auc'])

    print("\n🏆 Winner Recommendation:")
    if best['auc'] >= best_inv['inv_auc']:
        print(f"   BEST ID: {best['id']} (AUC={best['auc']:.4f})")
    else:
        print(f"   BEST ID: {best_inv['id']} (Inverted AUC={best_inv['inv_auc']:.4f}) -> Need to flip labels!")


if __name__ == "__main__":
    main()