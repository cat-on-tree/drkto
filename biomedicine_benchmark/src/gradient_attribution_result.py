import pandas as pd

# 假设你的csv叫做data.csv
df = pd.read_csv('../data/evaluation/gradient_attribution/2_checkpoint-3500_grad.csv')

# 严格按 output_token_id, input_token_id 排序
df = df.sort_values(['output_token_id', 'input_token_id'])

# 获取所有的 input_decoded（顺序与input_token_id相同，不去重）
input_decoded_list = df.drop_duplicates(subset=['input_token_id', 'input_decoded']).sort_values('input_token_id')['input_decoded'].tolist()
input_token_id_list = df.drop_duplicates(subset=['input_token_id', 'input_decoded']).sort_values('input_token_id')['input_token_id'].tolist()

# 获取所有的 output_decoded（顺序与output_token_id相同，不去重）
output_decoded_list = df.drop_duplicates(subset=['output_token_id', 'output_decoded']).sort_values('output_token_id')['output_decoded'].tolist()
output_token_id_list = df.drop_duplicates(subset=['output_token_id', 'output_decoded']).sort_values('output_token_id')['output_token_id'].tolist()

# 构建矩阵
matrix = []
for out_id, out_dec in zip(output_token_id_list, output_decoded_list):
    row = [out_dec]
    for in_id in input_token_id_list:
        # 找到对应的gradient_attribution
        val = df[(df['output_token_id'] == out_id) & (df['input_token_id'] == in_id)]['gradient_attribution']
        row.append(val.values[0] if not val.empty else "")
    matrix.append(row)

# 生成最终DataFrame，第一行为input_decoded，第一列为output_decoded
columns = [''] + input_decoded_list
final_df = pd.DataFrame(matrix, columns=columns)

# 保存为csv
final_df.to_csv('../data/evaluation/gradient_attribution/qwen3-8b-kto-result.csv', index=False)
