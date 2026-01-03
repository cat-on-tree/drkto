import os
import argparse
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_pure_prompt(sample):
    prompt = ""
    if 'instruction' in sample and sample['instruction'].strip():
        prompt += sample['instruction'].strip() + "\n"
    if 'input' in sample and sample['input'].strip():
        prompt += sample['input'].strip()
    return prompt

def decode_tokens(tokenizer, tokens):
    # tokens: a list of token strings
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return [tokenizer.decode([id]) for id in ids]

def compute_gradient_attribution_batch(model, tokenizer, prompt, max_new_tokens=512, batch_size=5):
    # 1. Tokenize and get input_ids
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(model.device)
    input_len = input_ids.shape[1]

    # 2. Generate output tokens
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'].to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = gen_output[0]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_text = gen_text[len(prompt):].strip()

    # Fix: convert ids to tokens before decode_tokens, for both input and output
    input_tokens = tokenizer.convert_ids_to_tokens(gen_ids[:input_len])
    output_ids = gen_ids[input_len:]
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids)

    input_labels = decode_tokens(tokenizer, input_tokens)
    output_labels = decode_tokens(tokenizer, output_tokens)

    output_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in output_tokens]
    output_len = len(output_tokens)

    grad_matrix = []
    # 分批处理
    for batch_start in range(0, output_len, batch_size):
        batch_end = min(batch_start + batch_size, output_len)
        for rel_out_idx, out_idx in enumerate(range(batch_start, batch_end)):
            # 当前output token在gen_ids中的真实索引
            real_out_idx = input_len + out_idx

            curr_ids = gen_ids[:real_out_idx+1].unsqueeze(0).clone().detach().to(model.device)
            embedding_layer = model.get_input_embeddings()
            embeds = embedding_layer(curr_ids)
            embeds.retain_grad()
            embeds.requires_grad_()

            outputs = model(inputs_embeds=embeds)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            target_logits = logits[0, -1]  # Only last token
            target_token_id = tokenizer.convert_tokens_to_ids(output_tokens[out_idx])
            target_prob = torch.softmax(target_logits, dim=0)[target_token_id]
            model.zero_grad()
            if embeds.grad is not None:
                embeds.grad.zero_()
            target_prob.backward()
            grads = embeds.grad[0, :input_len]  # (input_len, embed_dim)
            attr = grads.norm(p=1, dim=-1).detach().cpu().numpy()
            grad_matrix.append(attr)
            # 显存清理
            del embeds, outputs, logits, target_logits, target_prob, grads
            torch.cuda.empty_cache()
    grad_matrix = torch.tensor(grad_matrix)  # (output_len, input_len)

    # Prepare CSV
    csv_rows = []
    for i, out_tok in enumerate(output_tokens):
        for j, in_tok in enumerate(input_tokens):
            csv_rows.append({
                "output_token_id": i,
                "output_token": out_tok,
                "output_decoded": output_labels[i],
                "input_token_id": j,
                "input_token": in_tok,
                "input_decoded": input_labels[j],
                "gradient_attribution": float(grad_matrix[i,j])
            })
    return gen_text, csv_rows, grad_matrix, input_tokens, output_tokens, input_labels, output_labels

def save_gradient_heatmap(grad_matrix, input_labels, output_labels, save_path, title="Gradient Attribution Heatmap"):
    if grad_matrix.shape[0] == 0 or grad_matrix.shape[1] == 0:
        print(f"Heatmap not generated: empty matrix or tokens.")
        return
    plt.figure(figsize=(min(12, len(input_labels)*0.7), min(8, len(output_labels)*0.7)))
    plt.imshow(grad_matrix, aspect='auto', cmap='Reds')
    plt.colorbar(label='Gradient Attribution')
    plt.xticks(range(len(input_labels)), input_labels, rotation=90, fontsize=6)
    plt.yticks(range(len(output_labels)), output_labels, fontsize=6)
    plt.xlabel('Input Tokens (decoded)')
    plt.ylabel('Output Tokens (decoded)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Qwen3梯度归因分批处理分析与可视化。")
    parser.add_argument('--input', type=str, default='./gradient_attribution/input', help="Directory containing question txt files (JSON format).")
    parser.add_argument('--output', type=str, default='./gradient_attribution/output', help="Directory to save answer txt files.")
    parser.add_argument('--csv', type=str, default='./gradient_attribution/csv', help="Directory to save gradient attribution csv files.")
    parser.add_argument('--jpg', type=str, default='./gradient_attribution/jpg', help="Directory to save gradient attribution jpg files.")
    parser.add_argument('--model', type=str, required=True, help="Path to local Huggingface model.")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for gradient attribution (output tokens per batch).")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Max output tokens to generate.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.csv, exist_ok=True)
    os.makedirs(args.jpg, exist_ok=True)

    model_name = os.path.basename(args.model.rstrip('/'))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda")
    model.eval()

    for filename in os.listdir(args.input):
        if filename.endswith(".txt"):
            filepath = os.path.join(args.input, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    sample = json.load(f)
                prompt = build_pure_prompt(sample)
                answer, grad_csv, grad_matrix, input_tokens, output_tokens, input_labels, output_labels = compute_gradient_attribution_batch(
                    model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
                )
                base = os.path.splitext(filename)[0]
                answer_path = os.path.join(args.output, f"{base}_{model_name}_answer.txt")
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(answer)
                csv_path = os.path.join(args.csv, f"{base}_{model_name}_grad.csv")
                pd.DataFrame(grad_csv).to_csv(csv_path, index=False)
                jpg_path = os.path.join(args.jpg, f"{base}_{model_name}_grad.jpg")
                save_gradient_heatmap(grad_matrix, input_labels, output_labels, jpg_path, title="Gradient Attribution Heatmap")
                print(f"Processed {filename} -> {answer_path}, {csv_path}, {jpg_path}")
            except Exception as e:
                import traceback
                print(f"Error processing {filename}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()