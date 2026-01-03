import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def chat_completion_with_stream(client, **kwargs):
    """
    调用流式接口，返回完整内容
    """
    stream_options = kwargs.pop("stream_options", None)
    # Qwen平台建议加上if chunk.choices判断
    completion = client.chat.completions.create(stream=True, **kwargs)
    full_content = ""
    for chunk in completion:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_content += delta.content
    return full_content

def chat_completion_with_sync(client, **kwargs):
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content.strip()

def process_item(idx, obj, args, client):
    max_retries = args.max_retries
    retry_base_wait = args.retry_base_wait

    if "idx" not in obj:
        msg = "原始数据缺少idx字段，跳过。"
        logging.warning(msg)
        return {"error": msg}

    out_idx = obj["idx"]
    question = obj.get("question", "")
    raw_context = obj.get("text", "")
    supporting_sentences = obj.get("supporting_sentences", [])
    context = clean_context(raw_context, supporting_sentences=supporting_sentences)

    # 第一轮
    messages1 = [{"role": "system", "content": system_prompt_round1()}]
    user_prompt1 = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Please answer the question directly and concisely."
    )
    messages1.append({"role": "user", "content": user_prompt1})

    for attempt in range(max_retries):
        try:
            kwargs = dict(model=args.model, messages=messages1)
            # 判断是否需要流式
            if args.enable_thinking_round1 is not None:
                kwargs["extra_body"] = {"enable_thinking": args.enable_thinking_round1}
                if args.enable_thinking_round1:
                    answer1 = chat_completion_with_stream(client, **kwargs)
                else:
                    answer1 = chat_completion_with_sync(client, **kwargs)
            else:
                answer1 = chat_completion_with_sync(client, **kwargs)
            answer1 = answer1.strip()
            break
        except Exception as e:
            wait_time = retry_base_wait * (2 ** attempt)
            errmsg = f"第{out_idx}条(Q1)请求出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
            print(errmsg)
            logging.warning(errmsg)
            time.sleep(wait_time)
    else:
        errmsg = f"第{out_idx}条(Q1)请求连续{max_retries}次失败，已跳过。最后错误：{e}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    # 第二轮
    messages2 = [{"role": "system", "content": system_prompt_round2()}]
    user_prompt2 = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer1}\n"
        "Please quote the supporting sentence from the context."
    )
    messages2.append({"role": "user", "content": user_prompt2})

    for attempt in range(max_retries):
        try:
            kwargs = dict(model=args.model, messages=messages2)
            if args.enable_thinking_round2 is not None:
                kwargs["extra_body"] = {"enable_thinking": args.enable_thinking_round2}
                if args.enable_thinking_round2:
                    answer2 = chat_completion_with_stream(client, **kwargs)
                else:
                    answer2 = chat_completion_with_sync(client, **kwargs)
            else:
                answer2 = chat_completion_with_sync(client, **kwargs)
            answer2 = answer2.strip()
            break
        except Exception as e:
            wait_time = retry_base_wait * (2 ** attempt)
            errmsg = f"第{out_idx}条(Q2)请求出错：{e}，第{attempt + 1}次重试，等待{wait_time:.1f}秒..."
            print(errmsg)
            logging.warning(errmsg)
            time.sleep(wait_time)
    else:
        errmsg = f"第{out_idx}条(Q2)请求连续{max_retries}次失败，已跳过。最后错误：{e}"
        print(errmsg)
        logging.error(errmsg)
        return {"idx": out_idx, "error": errmsg}

    llm_output = (
        f"Answer to Question 1: {answer1}\n"
        f"Answer to Question 2 (Supporting sentence): {answer2}"
    )

    logging.info(f"第{out_idx}条成功生成。")
    return {
        "idx": out_idx,
        "system_round1": system_prompt_round1(),
        "user_round1": user_prompt1,
        "system_round2": system_prompt_round2(),
        "user_round2": user_prompt2,
        "llm_output": llm_output
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--input", default="data/evaluation/benchmark/BioASQ.json", help="输入文件名")
    parser.add_argument("--output", required=True, help="输出文件名")
    parser.add_argument("--log", default="bioASQ-answer.log", help="日志文件名")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API KEY")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API base url")
    parser.add_argument("--threads", type=int, default=1, help="并发线程数，默认单线程")
    parser.add_argument("--max_retries", type=int, default=5, help="每条最大重试次数")
    parser.add_argument("--retry_base_wait", type=float, default=20, help="429出错后等待的基础秒数，指数退避")
    parser.add_argument("--enable_thinking_round1", nargs='?', const=False, default=None, help="第一轮enable_thinking，True/False，不传则None，只写参数名为False")
    parser.add_argument("--enable_thinking_round2", nargs='?', const=False, default=None, help="第二轮enable_thinking，True/False，不传则None，只写参数名为False")
    args = parser.parse_args()

    # 转换字符串为布尔值或None
    args.enable_thinking_round1 = str2bool_or_none(args.enable_thinking_round1)
    args.enable_thinking_round2 = str2bool_or_none(args.enable_thinking_round2)

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    items = load_items(args.input)

    # 断点续跑：统计已完成idx
    done_set = get_done_set(args.output)
    print(f"检测到已完成 {len(done_set)} 条，将跳过这些样本...")

    if args.threads == 1:
        with open(args.output, "a", encoding="utf-8") as fout:  # 追加模式
            for idx, obj in enumerate(tqdm(items, desc="LLM生成中")):
                if obj.get("idx") in done_set:
                    continue
                out = process_item(idx, obj, args, client)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成，结果保存在 {args.output}")

    else:
        to_process = [(idx, obj) for idx, obj in enumerate(items) if obj.get("idx") not in done_set]
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, idx, obj, args, client) for idx, obj in to_process]
            for future in tqdm(as_completed(futures), total=len(to_process), desc="LLM生成中(并发)"):
                out = future.result()
                if out is not None and "idx" in out:
                    results[out["idx"]] = out

        with open(args.output, "a", encoding="utf-8") as fout:  # 追加模式
            for idx in sorted(results):
                out = results[idx]
                if out is None:
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        logging.info(f"全部处理完成(并发)，结果保存在 {args.output}")

if __name__ == "__main__":
    main()