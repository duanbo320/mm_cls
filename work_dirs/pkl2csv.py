

import pandas as pd

# 读取 .pkl 文件
data_list = pd.read_pickle('result.pkl')

# 初始化一个空的 DataFrame
data_df = pd.DataFrame(columns=['img_path', 'pred_score_ill', 'pred_score_normal', 'gt_label', 'pred_label'])

# 遍历列表，将每个元素添加到 DataFrame
for i, item in enumerate(data_list):
    data_df.loc[i] = [item['img_path'], item['pred_score'][0].item(), item['pred_score'][1].item(), item['gt_label'].item(), item['pred_label'].item()]

# 将 data frame 转换为 .csv 文件
data_df.to_csv('result.csv', index=False)

