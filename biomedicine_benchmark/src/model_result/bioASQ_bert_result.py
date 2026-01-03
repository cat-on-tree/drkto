import json
import re
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import logging
from transformers import BertTokenizer

MAX_BERT_LEN = 512
tokenizer = BertTokenizer.from_pretrained("./model/bert-base-uncased")

def extract_gold_supporting_sentences(gold_item):
    return gold_item.get("supporting_sentences", [])

def extract_llm_supporting_sentence(llm_item):
    llm_output = llm_item.get("llm_output", "")
    match = re.search(r"Answer to Question 2 \(Supporting sentence\):\s*(.+)", llm_output, re.DOTALL)
    if match:
        line = match.group(1).split('\n')[0].strip()
        quote_match = re.search(r'["“](.+?)["”]', line)
        if quote_match:
            return quote_match.group(1)
        else:
            return line
    return ""

def choose_device():
    if torch.cuda.is_available():
        logging.info("Using GPU (cuda) for BERTScore evaluation.")
        return "cuda"
    elif torch.backends.mps.is_available():
        logging.info("Using Apple MPS for BERTScore evaluation.")
        return "mps"
    else:
        logging.info("Using CPU for BERTScore evaluation.")
        return "cpu"

def truncate_for_bert(text, max_len=MAX_BERT_LEN):
    # transformers会自动处理 special tokens，truncation=True 保证总长度不超过 max_len
    return tokenizer.decode(
        tokenizer.encode(
            text, add_special_tokens=True, max_length=max_len, truncation=True
        ),
        skip_special_tokens=True
    )

def main(gold_json_file, llm_json_file, report_file="supporting_sentence_report.txt"):
    logging.info(f"Loading gold file: {gold_json_file}")
    with open(gold_json_file, "r", encoding="utf-8") as f:
        gold_data = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Loading LLM answer file: {llm_json_file}")
    with open(llm_json_file, "r", encoding="utf-8") as f:
        llm_data = [json.loads(line) for line in f if line.strip()]

    gold_dict = {item["idx"]: item for item in gold_data}
    llm_dict = {item["idx"]: item for item in llm_data}

    gold_sents, pred_sents = [], []
    coverages, rouge_ls, rouge_1s, rouge_2s, bert_f1s = [], [], [], [], []
    skipped_indices = []

    logging.info(f"Starting evaluation on {len(set(gold_dict.keys()) & set(llm_dict.keys()))} samples.")

    # 只处理gold和llm都有非空supporting sentence的情况
    valid_indices = []
    for idx in sorted(set(gold_dict.keys()) & set(llm_dict.keys())):
        gold_item = gold_dict[idx]
        llm_item = llm_dict[idx]
        gold_support_list = extract_gold_supporting_sentences(gold_item)
        gold_support = gold_support_list[0].strip() if gold_support_list and gold_support_list[0].strip() else ""
        llm_support = extract_llm_supporting_sentence(llm_item).strip()
        if not gold_support or not llm_support:
            logging.warning(f"Skipped idx={idx}: gold or llm supporting sentence is empty. gold='{gold_support}', llm='{llm_support}'")
            skipped_indices.append(idx)
            continue
        # 加入截断
        gold_support_trunc = truncate_for_bert(gold_support)
        llm_support_trunc = truncate_for_bert(llm_support)
        gold_sents.append(gold_support_trunc)
        pred_sents.append(llm_support_trunc)
        valid_indices.append(idx)
        coverage = int(gold_support in llm_support or llm_support in gold_support)
        coverages.append(coverage)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gold_support, llm_support)
        rouge_1s.append(scores["rouge1"].fmeasure)
        rouge_2s.append(scores["rouge2"].fmeasure)
        rouge_ls.append(scores["rougeL"].fmeasure)
        logging.info(f"Evaluated idx={idx}: Coverage={coverage}, ROUGE-1={scores['rouge1'].fmeasure:.4f}, "
                     f"ROUGE-2={scores['rouge2'].fmeasure:.4f}, ROUGE-L={scores['rougeL'].fmeasure:.4f}")

    # 检查所有输入长度（可选，调试用，可注释掉）
    for i, sent in enumerate(pred_sents + gold_sents):
        ids = tokenizer.encode(sent, add_special_tokens=True)
        if len(ids) > MAX_BERT_LEN:
            logging.error(f"Truncated sentence still too long! idx={i}, length={len(ids)}")

    device = choose_device()
    logging.info("Starting BERTScore evaluation...")
    if gold_sents and pred_sents:
        P, R, F1 = bert_score(
            pred_sents,
            gold_sents,
            model_type="./model/bert-base-uncased",
            device=device,
            verbose=True,
            rescale_with_baseline=False,
            num_layers=12
        )
        bert_f1s = F1.tolist()
    else:
        bert_f1s = []
    logging.info("BERTScore evaluation finished.")

    with open(report_file, "w", encoding="utf-8") as fout:
        for i, idx in enumerate(valid_indices):
            fout.write(f"=== idx: {idx} ===\n")
            fout.write(f"Gold supporting: {gold_sents[i]}\n")
            fout.write(f"LLM supporting:  {pred_sents[i]}\n")
            fout.write(f"Coverage: {coverages[i]}\n")
            fout.write(f"ROUGE-1: {rouge_1s[i]:.4f} ROUGE-2: {rouge_2s[i]:.4f} ROUGE-L: {rouge_ls[i]:.4f}\n")
            fout.write(f"BERTScore-F1: {bert_f1s[i]:.4f}\n\n")
        fout.write("==== 总体指标 ====\n")
        fout.write(f"样本数: {len(gold_sents)}\n")
        fout.write(f"Coverage: {sum(coverages)/len(coverages):.4f}\n" if coverages else "Coverage: N/A\n")
        fout.write(f"ROUGE-1: {sum(rouge_1s)/len(rouge_1s):.4f}\n" if rouge_1s else "ROUGE-1: N/A\n")
        fout.write(f"ROUGE-2: {sum(rouge_2s)/len(rouge_2s):.4f}\n" if rouge_2s else "ROUGE-2: N/A\n")
        fout.write(f"ROUGE-L: {sum(rouge_ls)/len(rouge_ls):.4f}\n" if rouge_ls else "ROUGE-L: N/A\n")
        fout.write(f"BERTScore-F1: {sum(bert_f1s)/len(bert_f1s):.4f}\n" if bert_f1s else "BERTScore-F1: N/A\n")

    logging.info(f"评测完成，报告已保存到 {report_file}")
    if skipped_indices:
        logging.info(f"共跳过 {len(skipped_indices)} 个空句样本，索引为: {skipped_indices}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/BioASQ.json", help="金标准jsonl")
    parser.add_argument("--answer", required=True, help="llm生成数据jsonl")
    parser.add_argument("--result", default="bert_result.txt", help="报告输出文件")
    parser.add_argument("--log", default="bert_eval.log", help="日志文件路径（含文件名），如 logs/bert_eval.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log, encoding="utf-8")
        ]
    )

    main(args.benchmark, args.answer, args.result)