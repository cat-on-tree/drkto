import os
import json
import re
from tqdm import tqdm
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def clean_context(text, supporting_sentences=None):
    text = re.sub(r"<answer>.*?<context>", "", text, flags=re.DOTALL)
    text = re.sub(r"<answer>.*?(\n|$)", "", text, flags=re.DOTALL)
    text = text.replace("<context>", "")
    if supporting_sentences:
        for sent in supporting_sentences:
            sent = sent.strip()
            text = text.replace(sent, "")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def system_prompt_round1():
    return (
        "You are an expert in biomedicine, please read the context and answer the question. Here is an example:\n"
        "Context: JTV519 (K201) reduces sarcoplasmic reticulum Ca²⁺ leak and improves diastolic function in vitro in murine and human non-failing myocardium. BACKGROUND AND PURPOSE: Ca²⁺ leak from the sarcoplasmic reticulum (SR) via ryanodine receptors (RyR2s) contributes to cardiomyocyte dysfunction. RyR2 Ca²⁺ leak has been related to RyR2 phosphorylation. In these conditions, JTV519 (K201), a 1,4-benzothiazepine derivative and multi-channel blocker, stabilizes RyR2s and decrease SR Ca²⁺ leak. We investigated whether JTV519 stabilizes RyR2s without increasing RyR2 phosphorylation in mice and in non-failing human myocardium and explored underlying mechanisms. EXPERIMENTAL APPROACH: SR Ca²⁺ leak was induced by ouabain in murine cardiomyocytes. [Ca²⁺]-transients, SR Ca²⁺ load and RyR2-mediated Ca²⁺ leak (sparks/waves) were quantified, with or without JTV519 (1 µmol·L⁻¹). Contribution of Ca²⁺ -/calmodulin-dependent kinase II (CaMKII) was assessed by KN-93 and Western blot (RyR2-Ser(2814) phosphorylation). Effects of JTV519 on contractile force were investigated in non-failing human ventricular trabeculae. KEY RESULTS: Ouabain increased systolic and diastolic cytosolic [Ca²⁺](i) , SR [Ca²⁺], and SR Ca²⁺ leak (Ca²⁺ spark (SparkF) and Ca²⁺ wave frequency), independently of CaMKII and RyR-Ser(2814) phosphorylation. JTV519 decreased SparkF but also SR Ca²⁺ load. At matched SR [Ca²⁺], Ca²⁺ leak was significantly reduced by JTV519, but it had no effect on fractional Ca²⁺ release or Ca²⁺ wave propagation velocity. In human muscle, JTV519 was negatively inotropic at baseline but significantly enhanced ouabain-induced force and reduced its deleterious effects on diastolic function. CONCLUSIONS AND IMPLICATIONS: JTV519 was effective in reducing SR Ca²⁺ leak by specifically regulating RyR2 opening at diastolic [Ca²⁺](i) in the absence of increased RyR2 phosphorylation at Ser(2814) , extending the potential use of JTV519 to conditions of acute cellular Ca²⁺ overload.\n"
        "Question: The drug JTV519 is derivative of which group of chemical compounds?\n"
        "Your answers should be:\n"
        "Answer: 1,4-benzothiazepine\n"
    )

def system_prompt_round2():
    return (
        "You are an expert in biomedicine, please read the context and analyze, what sentence in the context support the answer? Here is an example:\n"
        "Context: JTV519 (K201) reduces sarcoplasmic reticulum Ca²⁺ leak and improves diastolic function in vitro in murine and human non-failing myocardium. BACKGROUND AND PURPOSE: Ca²⁺ leak from the sarcoplasmic reticulum (SR) via ryanodine receptors (RyR2s) contributes to cardiomyocyte dysfunction. RyR2 Ca²⁺ leak has been related to RyR2 phosphorylation. In these conditions, JTV519 (K201), a 1,4-benzothiazepine derivative and multi-channel blocker, stabilizes RyR2s and decrease SR Ca²⁺ leak. We investigated whether JTV519 stabilizes RyR2s without increasing RyR2 phosphorylation in mice and in non-failing human myocardium and explored underlying mechanisms. EXPERIMENTAL APPROACH: SR Ca²⁺ leak was induced by ouabain in murine cardiomyocytes. [Ca²⁺]-transients, SR Ca²⁺ load and RyR2-mediated Ca²⁺ leak (sparks/waves) were quantified, with or without JTV519 (1 µmol·L⁻¹). Contribution of Ca²⁺ -/calmodulin-dependent kinase II (CaMKII) was assessed by KN-93 and Western blot (RyR2-Ser(2814) phosphorylation). Effects of JTV519 on contractile force were investigated in non-failing human ventricular trabeculae. KEY RESULTS: Ouabain increased systolic and diastolic cytosolic [Ca²⁺](i) , SR [Ca²⁺], and SR Ca²⁺ leak (Ca²⁺ spark (SparkF) and Ca²⁺ wave frequency), independently of CaMKII and RyR-Ser(2814) phosphorylation. JTV519 decreased SparkF but also SR Ca²⁺ load. At matched SR [Ca²⁺], Ca²⁺ leak was significantly reduced by JTV519, but it had no effect on fractional Ca²⁺ release or Ca²⁺ wave propagation velocity. In human muscle, JTV519 was negatively inotropic at baseline but significantly enhanced ouabain-induced force and reduced its deleterious effects on diastolic function. CONCLUSIONS AND IMPLICATIONS: JTV519 was effective in reducing SR Ca²⁺ leak by specifically regulating RyR2 opening at diastolic [Ca²⁺](i) in the absence of increased RyR2 phosphorylation at Ser(2814) , extending the potential use of JTV519 to conditions of acute cellular Ca²⁺ overload.\n"
        "Question: The drug JTV519 is derivative of which group of chemical compounds?\n"
        "Answer: 1,4-benzothiazepine\n"
        "your analyze should be: \n"
        "Supporting sentence: In these conditions, JTV519 (K201), a 1,4-benzothiazepine derivative and multi-channel blocker, stabilizes RyR2s and decrease SR Ca²⁺ leak.\n"
    )

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
    match = re.search(r"(?:^|\n)assistant\s*\n(.*)", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", result, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return result.strip()

def remove_think_tags(text):
    return re.sub(r"<think>\s*?</think>\s*", "", text, flags=re.DOTALL)

def local_generate(
    model, tokenizer, system_prompt, user_prompt, max_new_tokens=4096, device="cpu",
    enable_thinking=None, use_qwen_template=False, stream=False, temperature=1.0, top_k=50
):
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
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_result = extract_assistant_response(result)
    assistant_result = remove_think_tags(assistant_result)
    return assistant_result

def get_done_set(output_path):
    done_set = set()
    if not os.path.exists(output_path):
        return done_set
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "idx" in obj:
                    done_set.add(obj["idx"])
            except Exception:
                continue
    return done_set

def str2bool_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地transformers模型目录")
    parser.add_argument("--input", default="data/evaluation/benchmark/BioASQ.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="BioASQ-answer.log", help="日志文件名")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 mps")
    parser.add_argument("--threads", type=int, default=1, help="并发线程数，默认单线程")
    parser.add_argument("--enable_thinking_round1", nargs='?', const=False, default=None, help="第一轮enable_thinking，True/False，不传则None，只写参数名为False")
    parser.add_argument("--enable_thinking_round2", nargs='?', const=False, default=None, help="第二轮enable_thinking，True/False，不传则None，只写参数名为False")
    parser.add_argument("--stream", action="store_true", help="强制流式解码（如模型支持），否则仅在enable_thinking为True时自动流式")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="采样top_k")
    args = parser.parse_args()

    args.enable_thinking_round1 = str2bool_or_none(args.enable_thinking_round1)
    args.enable_thinking_round2 = str2bool_or_none(args.enable_thinking_round2)

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    items = load_items(args.input)
    done_set = get_done_set(args.output)
    print(f"检测到已完成 {len(done_set)} 条，将跳过这些样本...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if device == "mps":
        model = model.to(torch.float32)
    model = model.to(device).eval()

    model_type = getattr(model.config, "model_type", "").lower()
    use_qwen_template = "qwen" in model_type

    with open(args.output, "a", encoding="utf-8") as fout:
        for idx, obj in enumerate(tqdm(items, desc="本地模型推理中")):
            if "idx" not in obj:
                msg = "原始数据缺少idx字段，跳过。"
                logging.warning(msg)
                continue
            out_idx = obj["idx"]
            if out_idx in done_set:
                continue
            question = obj.get("question", "")
            raw_context = obj.get("text", "")
            supporting_sentences = obj.get("supporting_sentences", [])
            context = clean_context(raw_context, supporting_sentences=supporting_sentences)

            # 第一轮
            user_prompt1 = f"Context: {context}\nQuestion: {question}\nPlease answer the question directly and concisely."
            try:
                answer1 = local_generate(
                    model, tokenizer,
                    system_prompt_round1(), user_prompt1,
                    max_new_tokens=args.max_new_tokens, device=device,
                    enable_thinking=args.enable_thinking_round1,
                    use_qwen_template=use_qwen_template,
                    stream=args.stream or (args.enable_thinking_round1 is True),
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                answer1 = answer1.strip()
                logging.info(f"第{out_idx}条Q1本地模型生成成功。")
            except Exception as e:
                errmsg = f"第{out_idx}条Q1本地模型推理出错：{e}"
                print(errmsg)
                logging.error(errmsg)
                continue

            # 第二轮
            user_prompt2 = f"Context: {context}\nQuestion: {question}\nAnswer: {answer1}\nPlease quote the supporting sentence from the context."
            try:
                answer2 = local_generate(
                    model, tokenizer,
                    system_prompt_round2(), user_prompt2,
                    max_new_tokens=args.max_new_tokens, device=device,
                    enable_thinking=args.enable_thinking_round2,
                    use_qwen_template=use_qwen_template,
                    stream=args.stream or (args.enable_thinking_round2 is True),
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                answer2 = answer2.strip()
                logging.info(f"第{out_idx}条Q2本地模型生成成功。")
            except Exception as e:
                errmsg = f"第{out_idx}条Q2本地模型推理出错：{e}"
                print(errmsg)
                logging.error(errmsg)
                continue

            llm_output = (
                f"Answer to Question 1: {answer1}\n"
                f"Answer to Question 2 (Supporting sentence): {answer2}"
            )

            out = {
                "idx": out_idx,
                "system_round1": system_prompt_round1(),
                "user_round1": user_prompt1,
                "system_round2": system_prompt_round2(),
                "user_round2": user_prompt2,
                "llm_output": llm_output
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    logging.info(f"全部处理完成，结果保存在 {args.output}")

if __name__ == "__main__":
    main()