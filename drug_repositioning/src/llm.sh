#modelscope download --model Qwen/qwen3-8b --local_dir ../model/qwen3-8b

python llm_local_8bit.py \
  --input_file "../data/benchmark/Kaggle_drug_repositioning/test_llm.csv" \
  --output_jsonl "../data/evaluation/llm_prob/qwen3-kto.jsonl" \
  --model_path "../model/qwen3-kto"

python llm_plot.py \
  --input_jsonl "../data/evaluation/llm_prob/qwen3-kto.jsonl" \
  --output_dir "../data/evaluation/llm_plot"

python llm_logprob.py \
  --input_file "../data/benchmark/Kaggle_drug_repositioning/test_llm.csv" \
  --output_jsonl "../data/evaluation/llm_prob/qwen3-235b-a22b.jsonl" \
  --model "qwen3-235b-a22b"

