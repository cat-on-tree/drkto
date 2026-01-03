import os
import json
import re
from tqdm import tqdm
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_prompts(text):
    sys_match = re.search(r'(TASK:.*?Example-1 A: false)', text, re.DOTALL)
    user_matches = re.findall(r'(Q:.*?)(?=(?:\n\nQ:|\Z))', text, re.DOTALL)
    if sys_match and user_matches:
        system_prompt = sys_match.group(1).strip()
        user_prompt = user_matches[-1].strip()
        return system_prompt, user_prompt
    return None, None

def remove_think_tags(text):
    # 支持多组<think>...</think>，并允许换行
    return re.sub(r"<think>\s*?</think>\s*", "", text, flags=re.DOTALL)

def load_items(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def resolve_device(device_str):
    device_str = device_str.lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_str == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif device_str == "cpu":
        return "cpu"
    else:
        print(f"Warning: Requested device '{device_str}' not available. Falling back to CPU.")
        return "cpu"

def extract_assistant_response(result):
    """
    用正则从输出中提取assistant后的内容，去掉冗余的system/user段。
    支持常见三种格式（assistant、ASSISTANT、<|im_start|>assistant）。
    若未命中，则返回原文。
    """
    match = re.search(r"(?:^|\n)assistant\s*\n(.*)", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return result.strip()

def local_generate(model, tokenizer, system_prompt, user_prompt, max_new_tokens=4096, device="cpu", enable_thinking=None, use_qwen_template=False):
    # Qwen/chat模板兼容
    if use_qwen_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_template_kwargs = dict(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(**chat_template_kwargs)
    else:
        prompt = system_prompt + "\n" + user_prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_result = extract_assistant_response(result)
    assistant_result = remove_think_tags(assistant_result)
    return assistant_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地transformers模型目录")
    parser.add_argument("--input", default="data/evaluation/benchmark/chemprot.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="chemprot-answer.log", help="日志文件名")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 mps")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="出现此参数时，显式关闭思考模式（即传 enable_thinking=False），不传则保持模型默认")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    items = load_items(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if device == "mps":
        model = model.to(torch.float32)
    model = model.to(device).eval()

    # 判断是否Qwen系列（兼容大小写）
    model_type = getattr(model.config, "model_type", "").lower()
    use_qwen_template = "qwen" in model_type

    # enable_thinking 逻辑
    if args.enable_thinking:
        enable_thinking = False
    else:
        enable_thinking = None

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, obj in enumerate(tqdm(items, desc="本地模型推理中")):
            if "idx" not in obj:
                msg = "原始数据缺少idx字段，跳过。"
                logging.warning(msg)
                continue
            out_idx = obj["idx"]
            text = obj["unprocessed"]
            sys_prompt, user_prompt = extract_prompts(text)
            if not (sys_prompt and user_prompt):
                msg = f"第{out_idx + 1}条未能正确抽取prompt，跳过。"
                logging.warning(msg)
                continue
            try:
                result = local_generate(
                    model, tokenizer, sys_prompt, user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    enable_thinking=enable_thinking,
                    use_qwen_template=use_qwen_template
                )
                logging.info(f"第{out_idx + 1}条本地模型生成成功。")
                out = {
                    "idx": out_idx,
                    "system": sys_prompt,
                    "user": user_prompt,
                    "llm_output": result
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                errmsg = f"第{out_idx + 1}条本地模型推理出错：{e}"
                print(errmsg)
                logging.error(errmsg)
                continue
    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()