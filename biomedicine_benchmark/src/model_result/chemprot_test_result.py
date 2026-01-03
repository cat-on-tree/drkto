import json
import re
from sklearn.metrics import classification_report
import argparse

def extract_label_from_model(label_json_str):
    try:
        if isinstance(label_json_str, dict):
            return label_json_str.get("label")
        elif isinstance(label_json_str, str):
            return json.loads(label_json_str).get("label")
    except Exception:
        match = re.search(r'"label"\s*:\s*"([^"]+)"', label_json_str)
        if match:
            return match.group(1)
    return None

def main(model_file, gold_file, report_file):
    idx2pred = {}
    with open(model_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj["idx"]
            if "label_json" in obj:
                pred_label = extract_label_from_model(obj["label_json"])
            else:
                pred_label = None
            idx2pred[idx] = pred_label

    idx2gold = {}
    with open(gold_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj["idx"]
            gt_label = obj.get("processed", None)
            idx2gold[idx] = gt_label

    # 只评测有答案的部分
    common_idx = sorted(set(idx2pred.keys()) & set(idx2gold.keys()))
    preds = [idx2pred[i] for i in common_idx]
    gts = [idx2gold[i] for i in common_idx]

    # 统计所有出现过的label
    labels = sorted(list(set(gts) | set(preds)))
    # 有的None, 过滤掉
    y_true = [g if g is not None else "None" for g in gts]
    y_pred = [p if p is not None else "None" for p in preds]

    report = classification_report(
        y_true, y_pred, labels=labels, digits=2, zero_division=0
    )

    print("==== Classification Report ====")
    print(report)

    with open(report_file, "w", encoding="utf-8") as fout:
        fout.write("==== Classification Report ====\n")
        fout.write(report)
    print(f"\n分类报告已保存到 {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="模型输出jsonl文件")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/chemprot.json", help="金标准jsonl文件")
    parser.add_argument("--result", default="classification_report.txt", help="评测报告输出txt文件")
    args = parser.parse_args()
    main(args.test, args.benchmark, args.result)