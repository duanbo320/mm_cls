import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 从CSV文件中读取数据
data = pd.read_csv('result.csv')

# 将列数据转为Python列表
pred_score_normal = data['pred_score_normal'].tolist()
gt_label = data['gt_label'].tolist()
pred_label = data['pred_label'].tolist()

# 计算roc曲线
fpr, tpr, _ = roc_curve(gt_label, pred_score_normal)

roc_auc = auc(fpr, tpr)

# 绘制roc曲线
plt.figure(figsize=(5, 5), dpi=300)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

# 保存ROC曲线图像
plt.savefig('roc_curve.png')

# 显示ROC曲线
plt.show()

# 计算混淆矩阵
cm = confusion_matrix(gt_label, pred_label)

# 绘制混淆矩阵
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = ''
plt.title(all_sample_title, size=15)

# 保存混淆矩阵图像
plt.savefig('confusion_matrix.png')

# 显示混淆矩阵
plt.show()


import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 从CSV文件中读取数据
data = pd.read_csv('result.csv')

# 将列数据转为Python列表
gt_label = data['gt_label'].tolist()
pred_label = data['pred_label'].tolist()

# 计算混淆矩阵
cm = confusion_matrix(gt_label, pred_label)

# 计算精确度（precision）
precision = precision_score(gt_label, pred_label)

# 计算召回率（recall）
recall = recall_score(gt_label, pred_label)

# 计算F1-score
f1score = f1_score(gt_label, pred_label)

# 计算准确率（accuracy）
accuracy = accuracy_score(gt_label, pred_label)

# 创建包含结果的DataFrame
results = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-score': [f1score]
})

# 保存结果到CSV文件
results.to_csv('metrics.csv', index=False)

import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

n_iterations = 1000  # 可以改为您需要的迭代次数
n_size = len(pred_score_normal)  # 您的样本数量

# 这个方法应该返回计算ROC曲线所需的fpr, tpr和thresholds
def compute_roc_auc(index):
    # 使用您自己计算ROC曲线的逻辑替换这一部分
    # 这里只是一个例子
    y_true = np.array(gt_label)[index]
    y_score = np.array(pred_score_normal)[index]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, _

# 获取所有可能的fpr值
all_fpr = np.unique(np.concatenate([fpr for fpr, tpr, _ in [compute_roc_auc(range(n_size)) for _ in range(n_iterations)]]))

# 初始化一个数组来保存插值的tpr值
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_iterations):
    index = resample(range(n_size), n_samples=n_size, replace=True)
    fpr, tpr, _ = compute_roc_auc(index)
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= n_iterations

# 计算95%置信区间
all_tpr_interp = np.zeros((n_iterations, len(all_fpr)))
for i in range(n_iterations):
    index = resample(range(n_size), n_samples=n_size, replace=True)
    fpr, tpr, _ = compute_roc_auc(index)
    all_tpr_interp[i] = np.interp(all_fpr, fpr, tpr)

std_tpr = np.std(all_tpr_interp, axis=0)
tpr_upper = np.minimum(mean_tpr + 1.96*std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - 1.96*std_tpr, 0)

# 绘制ROC曲线和置信区间
plt.plot(all_fpr, mean_tpr, color='b', label='Mean ROC')
plt.fill_between(all_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='95% CI')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# 保存图片
plt.savefig('mean_roc_with_ci_.png')
plt.show()


# 计算95%置信区间的范围
# 注意: 请确保您已经正确计算了tpr_lower和tpr_upper
ci_lower = np.quantile(tpr_lower, 0.025)  # 2.5分位数
ci_upper = np.quantile(tpr_upper, 0.975)  # 97.5分位数

# 绘制ROC曲线和置信区间
plt.plot(all_fpr, mean_tpr, color='b', label=f'Mean ROC\n95% CI = ({ci_lower:.2f}, {ci_upper:.2f})')
plt.fill_between(all_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 设置legend
plt.legend(loc='lower right')

# 保存图片
plt.savefig('mean_roc_with_ci_and_annotation.png')

# 显示图片
plt.show()