import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_answer_context_gold(item):
    """
    从benchmark每条item中提取问题、参考文本、金标准答案、金标准支持句
    """
    question = item.get("question", "")
    text = item.get("text", "")
    answer_match = re.search(r"<answer>\s*([^\n<]+)", text)
    context_match = re.search(r"<context>\s*([^\n]+)", text)
    gold_answer = answer_match.group(1).strip() if answer_match else ""
    context = context_match.group(1).strip() if context_match else ""
    supporting_sentences = item.get("supporting_sentences", [])
    gold_support = supporting_sentences[0].strip() if supporting_sentences else ""
    return question, context, gold_answer, gold_support

def extract_llm_answer_and_support(llm_output):
    """
    从llm输出中提取模型回答和支持句
    """
    answer_match = re.search(r"Answer to Question 1:(.*?)(?:Answer to Question 2|\Z)", llm_output, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    support_match = re.search(r"Answer to Question 2.*?:\s*(.*)", llm_output, re.DOTALL)
    supporting_sentence = support_match.group(1).strip() if support_match else ""
    return answer, supporting_sentence

def build_prompt(question, context, gold_answer, gold_support, model_answer, model_support):
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Model Supporting Sentence:\n{model_support}\n\n"
        f"Gold Answer:\n{gold_answer}\n\n"
        f"Gold Supporting Sentence:\n{gold_support}\n\n"
        f"Please rate the model's answer and supporting sentence following the instructions below and explain your score in JSON format."
    )

def score_one(idx, question, context, gold_answer, gold_support, model_answer, model_support, system_prompt, client, model, max_retries=5):
    user_prompt_full = build_prompt(question, context, gold_answer, gold_support, model_answer, model_support)
    wait_times = [5, 10, 15, 20, 25]
    last_exception = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_full},
                ],
                response_format={"type": "json_object"},
            )
            result = completion.choices[0].message.content
            try:
                json.loads(result)
            except Exception as e:
                logging.error(f"idx={idx} LLM输出不是合法JSON: {e}")
                return {
                    "idx": idx,
                    "error": f"LLM输出不是合法JSON: {e}",
                    "gold_answer": gold_answer,
                    "gold_support": gold_support,
                    "model_answer": model_answer,
                    "model_support": model_support,
                    "retries": attempt+1
                }
            return {
                "idx": idx,
                "gptscore_json": result,
                "gold_answer": gold_answer,
                "gold_support": gold_support,
                "model_answer": model_answer,
                "model_support": model_support,
                "retries": attempt+1
            }
        except Exception as e:
            last_exception = e
            logging.warning(f"idx={idx} 第{attempt+1}次请求出错: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_times[attempt])
    errmsg = f"idx={idx} 请求重试{max_retries}次仍失败: {last_exception}"
    logging.error(errmsg)
    return {
        "idx": idx,
        "error": errmsg,
        "gold_answer": gold_answer,
        "gold_support": gold_support,
        "model_answer": model_answer,
        "model_support": model_support,
        "retries": max_retries
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="评价用的大模型名称")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark/BioASQ.json", help="输入benchmark.json文件名")
    parser.add_argument("--answer", required=True, help="待评价答案的jsonl文件名")
    parser.add_argument("--output", required=True, help="输出jsonl文件名")
    parser.add_argument("--log", default="bioASQ_gptscore_test.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数量")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 放宽标准的few-shot评分示例
    GOLD_ANSWER = "glucocerebrosidase"
    GOLD_SUPPORT = "Gaucher's disease (GD) results from a deficiency of the lysosomal enzyme glucocerebrosidase and, in very rare occasions, a deficiency of its activator, the saposin C."

    example_5 = json.dumps({
        "answer": "The enzyme deficient in Gaucher's disease is glucocerebrosidase.",
        "supporting_sentence": "Gaucher's disease (GD) results from a deficiency of the lysosomal enzyme glucocerebrosidase and, in very rare occasions, a deficiency of its activator, the saposin C.",
        "score": 5,
        "reason": "The model answer is semantically equivalent to the gold answer and clearly expresses the complete meaning. The supporting sentence is directly extracted from context and fully covers the gold supporting sentence, including all key details. The logical connection between answer and supporting sentence is strong.",
        "match": True
    }, ensure_ascii=False)

    example_5_rich = json.dumps({
        "answer": "Gaucher's disease is caused by a deficiency of the lysosomal enzyme glucocerebrosidase, which leads to lipid accumulation within cells. In rare cases, deficiency of its activator, saposin C, may also be responsible. Glucocerebrosidase normally helps break down glucocerebroside, and its deficiency results in characteristic clinical manifestations.",
        "supporting_sentence": "Gaucher's disease results from a deficiency of the lysosomal enzyme glucocerebrosidase and, in very rare occasions, a deficiency of its activator, the saposin C. Glucocerebrosidase acts to cleave glucocerebroside in the lysosome, and its deficiency leads to lipid accumulation.",
        "score": 5,
        "reason": "The model answer is semantically equivalent to the gold answer and provides additional scientifically accurate details about the enzyme's function and the consequences of its deficiency. The supporting sentence covers all content of the gold supporting sentence and reasonably supplements it with mechanism and context, forming a stronger logical connection and richer information.",
        "match": True
    }, ensure_ascii=False)

    example_4 = json.dumps({
        "answer": "Gaucher's disease is caused by deficiency of the lysosomal enzyme glucocerebrosidase.",
        "supporting_sentence": "Gaucher's disease (GD) results from a deficiency of the lysosomal enzyme glucocerebrosidase.",
        "score": 4,
        "reason": "The model answer is semantically equivalent to the gold answer and provides a complete explanation. The supporting sentence is directly extracted from context and covers the main part of the gold supporting sentence but omits the detail about saposin C, so coverage is incomplete.",
        "match": True
    }, ensure_ascii=False)

    example_3 = json.dumps({
        "answer": "The gene mutated in Gaucher's disease is glucocerebrosidase, but sometimes saposin C may also be involved.",
        "supporting_sentence": "Gaucher's disease (GD) results from a deficiency of the lysosomal enzyme glucocerebrosidase and sometimes saposin C may be deficient.",
        "score": 3,
        "reason": "The model answer is close in meaning to the gold answer but is verbose and mixes relevant and irrelevant content. The supporting sentence partially overlaps with the gold supporting sentence, but includes some generated information not present in the context.",
        "match": True
    }, ensure_ascii=False)

    example_2 = json.dumps({
        "answer": "saposin C",
        "supporting_sentence": "Gaucher's disease may also result from a deficiency of saposin C.",
        "score": 2,
        "reason": "The model answer is not semantically equivalent to the gold answer but is present in context. The supporting sentence covers only a small part of the gold supporting sentence and does not logically support the main answer for the question.",
        "match": False
    }, ensure_ascii=False)

    example_1 = json.dumps({
        "answer": "Gaucher disease is a lysosomal storage disorder.",
        "supporting_sentence": "It is a lysosomal disorder. Mutations may be complex.",
        "score": 1,
        "reason": "The model answer and supporting sentence are irrelevant or mainly generated content, not extracted from context, and do not logically support the gold answer.",
        "match": False
    }, ensure_ascii=False)

    system_prompt = f"""
    You are an expert biomedical information extractor.
    Given a question, a context (background knowledge), a model answer, a supporting sentence, a gold answer, and a gold supporting sentence, rate the model's answer and supporting sentence according to the following criteria:

    Scoring reference:
    - 5: The model answer is semantically equivalent to the gold answer and provides additional, relevant, and accurate details or explanations that go beyond the gold answer, enhancing the scientific or clinical value of the response. The supporting sentence not only fully covers the gold supporting sentence, but also reasonably supplements it with related mechanisms, effects, or context, forming a stronger logical connection and richer information.
    - 4: The model answer is semantically equivalent to the gold answer and is complete, but does not provide additional relevant details or scientific explanations beyond the gold answer. The supporting sentence covers the main part of the gold supporting sentence, with limited or minor supplemental detail, or contains some unnecessary redundancy.
    - 3: The model answer is close in meaning to the gold answer, but is either verbose with irrelevant information or lacks key details. The supporting sentence partially covers the gold supporting sentence, may include generated or loosely related information, and provides limited logical support.
    - 2: The model answer is not semantically equivalent to the gold answer but is present in context. The supporting sentence covers little of the gold supporting sentence and does not logically support the answer; may be mostly generated content.
    - 1: The model answer and supporting sentence are irrelevant, incorrect, or mainly generated content (not from context), and do not logically support the gold answer.

    Special notes:
    - For answer: For 5 points, encourage model answers and supporting sentences that go beyond the gold standard in a scientifically valid way. Do not penalize answers for being lengthy or detailed if the extra information is correct and enhances the response. Penalize only for irrelevant or incorrect expansion.
    - For supporting sentence: Focus on how much of the gold supporting sentence is covered and whether it is directly extracted from context. Extra credit for scientifically relevant expansion.
    - Logical connection: Give extra credit if the supporting sentence logically supports both the question and the model answer. Deduct points if this logical connection is weak or absent.

    The gold answer and gold supporting sentence are for your reference only. DO NOT output or generate them in your result.

    Output JSON fields:
    - "score": integer, 1 (very poor) to 5 (excellent)
    - "reason": string, explaining your score, especially the reasoning evaluation
    - "match": boolean, true if the model answer is semantically equivalent to the gold answer, false otherwise

    Do NOT include the 'answer' or 'supporting_sentence' fields in your output JSON.
    The examples below include them for illustration only.

    Output format examples:
    {example_5}
    {example_5_rich}
    {example_4}
    {example_3}
    {example_2}
    {example_1}

    Please only output the JSON object with the specified fields.
    """

    # 读取benchmark数据（新结构）
    with open(args.benchmark, "r", encoding="utf-8") as fin:
        benchmark_items = [json.loads(line) for line in fin if line.strip()]
    idx2benchmark = {item["idx"]: item for item in benchmark_items}

    # 读取llm答案数据（新结构）
    with open(args.answer, "r", encoding="utf-8") as fin:
        answer_items = [json.loads(line) for line in fin if line.strip()]
    idx2answer = {item["idx"]: item for item in answer_items}

    # 构造所有任务
    tasks = []
    for cur_idx in sorted(idx2benchmark.keys()):
        benchmark_obj = idx2benchmark[cur_idx]
        answer_obj = idx2answer.get(cur_idx, None)
        question, context, gold_answer, gold_support = extract_answer_context_gold(benchmark_obj)
        llm_raw_output = answer_obj.get("llm_output", "") if answer_obj else ""
        model_answer, model_support = extract_llm_answer_and_support(llm_raw_output)
        tasks.append((cur_idx, question, context, gold_answer, gold_support, model_answer, model_support))

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future2idx = {
            executor.submit(
                score_one, idx, question, context, gold_answer, gold_support, model_answer, model_support,
                system_prompt, client, args.model, 5
            ): idx
            for idx, question, context, gold_answer, gold_support, model_answer, model_support in tasks
        }
        for future in tqdm(as_completed(future2idx), total=len(future2idx), desc="Concurrent LLM scoring w/ retry"):
            idx = future2idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = {
                    "idx": idx,
                    "error": f"Threaded error: {e}",
                    "gold_answer": "",
                    "gold_support": "",
                    "model_answer": "",
                    "model_support": "",
                    "retries": 5
                }
            results[idx] = out

    with open(args.output, "w", encoding="utf-8") as fout:
        for cur_idx in sorted(idx2benchmark.keys()):
            out = results.get(cur_idx, {"idx": cur_idx, "error": "No result generated.", "retries": 5})
            out["idx"] = cur_idx
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部评分处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()