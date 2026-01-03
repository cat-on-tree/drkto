import pandas as pd
import os


def check_ratio(file_path):
    print(f"========================================")
    print(f"📂 Analyzing: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)

        # 检查是否包含 'label' 列
        if 'label' not in df.columns:
            print(f"❌ Error: Column 'label' not found in {file_path}")
            print(f"   Available columns: {list(df.columns)}")
            return

        # 统计
        total = len(df)
        pos_count = len(df[df['label'] == 1])
        neg_count = len(df[df['label'] == 0])

        # 计算比例 (避免除以零)
        ratio = neg_count / pos_count if pos_count > 0 else 0

        print(f"   Total Samples: {total}")
        print(f"   ✅ Positive (1): {pos_count}")
        print(f"   ❌ Negative (0): {neg_count}")

        if pos_count > 0:
            print(f"   ⚖️ Ratio (Pos : Neg) = 1 : {ratio:.2f}")
        else:
            print(f"   ⚠️ Warning: No positive samples found!")

    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    # === 配置路径 (请根据你的实际路径修改) ===
    base_dir = "../data/benchmark/Kaggle_drug_repositioning/"

    files = [
        os.path.join(base_dir, "test.csv"),
        os.path.join(base_dir, "test_hard.csv"),
        os.path.join(base_dir, "test_cold.csv"),
        # 如果你有新的 cleaned 数据集，也可以加在这里
        # os.path.join(base_dir, "test_cold_llm_clean.csv")
    ]

    print("🚀 Starting Data Distribution Check...")

    for f in files:
        check_ratio(f)

    print(f"========================================")


if __name__ == "__main__":
    main()