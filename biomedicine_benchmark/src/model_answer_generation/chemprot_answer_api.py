import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_prompts(text):
    sys_match = re.search(r'(TASK:.*?Example-1 A: false)', text, re.DOTALL)
    user_matches = re.findall(r'(Q:.*?)(?=(?:\n\nQ:|\Z))', text, re.DOTALL)
    if sys_match and user_matches:
        system_prompt = sys_match.group(1).strip()
        user_prompt = user_matches[-1].strip()
        return system_prompt, user_prompt
    return None, None

def load_items(input_file):
    # 兼容扩展名为json但内容为jsonl的情况
    with open(input_file, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # 标准json数组
            return json.load(f)
        else:
            # 每行为一个json对象（jsonl）
            return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--input", default="data/evaluation/benchmark/chemprot.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="chemprot-answer.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=1, help="并发线程数，默认单线程")
    parser.add_argument("--max_retries", type=int, default=5, help="每条最大重试次数")
    parser.add_argument("--retry_base_wait", type=float, default=20, help="429出错后等待的基础秒数，指数退避")
    parser.add_argument("--enable_thinking", action="store_true", help="启用thinking模式（即请求时传extra_body={enable_thinking:False}）")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    items = load_items(args.input)
    # items = items[:10]
    def process_item(idx, obj):
        max_retries = args.max_retries
        retry_base_wait = args.retry_base_wait

        # 必须使用原始obj中的idx字段，没有则跳过
        if "idx" not in obj:
            msg = "原始数据缺少idx字段，跳过。"
            logging.warning(msg)
            return {"error": msg}

        out_idx = obj["idx"]

        text = obj["unprocessed"]
        sys_prompt, user_prompt = extract_prompts(text)
        if not (sys_prompt and user_prompt):
            msg = f"第{out_idx + 1}条未能正确抽取prompt，跳过。"
            logging.warning(msg)
            return {"idx": out_idx, "error": msg}
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                if args.enable_thinking:
                    kwargs["extra_body"] = {"enable_thinking": False}
                completion = client.chat.completions.create(**kwargs)

                result = completion.choices[0].message.content
                logging.info(f"第{out_idx + 1}条成功生成。")
                return {
                    "idx": out_idx,
                    "system": sys_prompt,
                    "user": user_prompt,
                    "llm_output": result
                }
            except Exception as e:
                wait_time = retry_base_wait * (2 ** attempt)
                errmsg = f"第{out_idx + 1}条请求出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
                print(errmsg)
                logging.warning(errmsg)
                time.sleep(wait_time)
                continue
        # 超过最大重试次数，记录错误
        errmsg = f"第{out_idx + 1}条请求连续{max_retries}次失败，已跳过。最后错误：{e}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    if args.threads == 1:
        with open(args.output, "w", encoding="utf-8") as fout:
            for idx, obj in enumerate(tqdm(items, desc="LLM生成中")):
                out = process_item(idx, obj)
                if "error" in out:
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fout.flush()
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成，结果保存在 {args.output}")

    else:
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, idx, obj) for idx, obj in enumerate(items)]
            for future in tqdm(as_completed(futures), total=len(items), desc="LLM生成中(并发)"):
                out = future.result()
                if out is not None and "idx" in out:
                    results[out["idx"]] = out

        with open(args.output, "w", encoding="utf-8") as fout:
            for idx in sorted(results):
                out = results[idx]
                if out is None:
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成(并发)，结果保存在 {args.output}")

if __name__ == "__main__":
    main()