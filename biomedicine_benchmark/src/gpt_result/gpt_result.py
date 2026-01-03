import argparse
import json
import re

def extract_scores(answer_file):
    score_list = []
    with open(answer_file, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line)
                gptscore_json = obj.get("gptscore_json", "")
                matches = re.findall(r'"score"\s*[:：]\s*([0-9]+)', gptscore_json, flags=re.MULTILINE)
                for m in matches:
                    score_list.append(int(m))
            except Exception as e:
                print(f"解析失败：{e}")
    return score_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="待评价答案的jsonl文件名")
    parser.add_argument("--result", required=True, help="输出txt文件名")
    args = parser.parse_args()

    score_list = extract_scores(args.test)
    lines = []
    if score_list:
        avg_score = sum(score_list) / len(score_list)
        lines.append(f"score_count: {len(score_list)}")
        lines.append(f"average_score: {avg_score:.4f}")
        for val in sorted(set(score_list)):
            lines.append(f"score_{val}_count: {score_list.count(val)}")
    else:
        lines.append("score_count: 0")
        lines.append("average_score: None")

    with open(args.result, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()