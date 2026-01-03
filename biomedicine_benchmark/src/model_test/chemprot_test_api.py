import os
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_one(idx, obj, client, args, max_retries=5):
    cur_idx = obj.get("idx", idx)
    system_prompt = (
        '''
        The task is to determine the relation description from a sentence, which concludes the relation between a chemical and a gene.
        Select one out of the six labels of relations ('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') without any explanation or other characters.
        The definations are as follows:
        CPR:3, which includes UPREGULATOR, ACTIVATOR, and INDIRECT UPREGULATOR;
        CPR:4, which includes DOWNREGULATOR, INHIBITOR ,and INDIRECT DOWNREGULATOR;
        CPR:5, which includes AGONIST, AGONIST ACTIVATOR, and AGONIST INHIBITOR;
        CPR:6, which includes ANTAGONIST;
        CPR:9, which includes SUBSTRATE, PRODUCT OF and SUBSTRATE PRODUCT OF;
        false, which indicates no relations.
        The sentence may contains the label directly, in that case, just output the label. Otherwise, determine the label according to the sentence.
        only output a json contains a label, the output format examples are as follows: 
        { "label": "CPR:3" }; { "label": "CPR:4" }; { "label": "false" }
        '''
    )
    user_prompt = obj.get("llm_output", "").strip()

    wait_times = [5, 10, 15, 20, 25]
    last_exception = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            result = completion.choices[0].message.content
            logging.info(f"第{cur_idx + 1}条成功生成。")
            out = {
                "idx": cur_idx,
                "label_json": result,
                "retries": attempt + 1
            }
            return cur_idx, out
        except Exception as e:
            last_exception = e
            errmsg = f"第{cur_idx + 1}条第{attempt+1}次请求出错：{e}"
            print(errmsg)
            logging.error(errmsg)
            if attempt < max_retries - 1:
                time.sleep(wait_times[attempt])
    out = {
        "idx": cur_idx,
        "error": f"请求重试{max_retries}次仍失败: {last_exception}",
        "retries": max_retries
    }
    return cur_idx, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-max-latest", help="模型名称")
    parser.add_argument("--input", required=True, help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="chemprot-test.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=4, help="并发线程数量")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 读取jsonl文件
    with open(args.input, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    idx2out = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_one, idx, obj, client, args)
            for idx, obj in enumerate(items)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Refine label by LLM (parallel)"):
            cur_idx, out = future.result()
            idx2out[cur_idx] = out

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in items:
            cur_idx = obj.get("idx")
            out = idx2out.get(cur_idx, {"idx": cur_idx, "error": "No result generated."})
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()