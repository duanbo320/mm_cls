import pandas as pd
import pickle

def convert_pkl_to_csv(pkl_path, csv_path):
    # 使用pickle读取pkl文件
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 遍历数据并添加到DataFrame
    for item in data:
        temp_df = pd.DataFrame([{
            'num_classes': item['num_classes'],
            'img_shape': item['img_shape'],
            'sample_idx': item['sample_idx'],
            'img_path': item['img_path'],
            'ori_shape': item['ori_shape'],
            'pred_score': item['pred_score'],
            'pred_label': item['pred_label'],
            'gt_label': item['gt_label']
        }])
        df = pd.concat([df, temp_df], ignore_index=True)

    # 保存为CSV文件
    df.to_csv(csv_path, index=False)

# 使用示例
pkl_file_path = 'result.pkl'  # 替换为您的pkl文件路径
csv_file_path = 'result_.csv'  # 替换为您希望保存的CSV文件路径

convert_pkl_to_csv(pkl_file_path, csv_file_path)
