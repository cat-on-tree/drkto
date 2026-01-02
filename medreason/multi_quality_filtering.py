import json
import utils
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_line(args):
    i, line, file_name = args
    try:
        sample = json.loads(line)
        question = sample['question']
        options = sample['options'] if 'options' in sample.keys() else ''
        answer = sample['answer']
        reasoning = sample['reasoning']
        reasoning = reasoning.split('Conclusion:')[0]

        dataset_name = file_name.split('.')[0].split('_')[0]
        logger = utils.init_logger(name=dataset_name)

        logger.info("Generating answer...")
        llm_answer = utils.llm_generate_answer_with_reasoning(
            question, options, reasoning, engine='qwen-max')
        logger.info(f'Q: {question}')
        logger.info(f'A: {answer}')
        logger.info(f'LLM-A: {llm_answer}')
        logger.info("Extracting answer...")
        logger.info("Judging answer...")
        judged_result = utils.llm_judge_answer(llm_answer, answer, engine='qwen-max')
        logger.info(f'Judged: {judged_result}')

        if "true" in judged_result.lower():
            logger.info(f'True sample {i} saved.')
            return line
        return None
    except Exception as e:
        # 你可以根据需要把错误信息记录到日志或打印
        print(f"Error processing line {i}: {e}")
        return None

def filter_file_parallel(file_name, num_workers=None):
    print("Start filtering for file: " + file_name)
    data_file = f'./results/filtered/{file_name}'
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return

    with open(data_file, 'r') as f:
        lines = f.readlines()

    target_dir = './results/quality_sampling'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_file_name = os.path.join(target_dir, file_name)

    args = [(i, lines[i], file_name) for i in range(len(lines))]
    with Pool(processes=(num_workers or cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_line, args), total=len(lines)))
    # 写入结果，只保留合格的
    with open(target_file_name, 'w') as f:
        for line in results:
            if line is not None:
                f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True,
                        help="待处理的jsonl文件名（位于./results/filtered/目录下）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行进程数，默认用CPU核心数")
    args = parser.parse_args()
    filter_file_parallel(args.source_file, args.num_workers)