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
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 从CSV文件中读取数据
data = pd.read_csv('result.csv')

# 将列数据转为Python列表
gt_label = data['gt_label'].tolist()
pred_label = data['pred_label'].tolist()

# 计算混淆矩阵
cm = confusion_matrix(gt_label, pred_label)

# 归一化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 绘制混淆矩阵
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(cm_normalized, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion Matrix', size=15)

# 保存混淆矩阵图像
plt.savefig('normalized_confusion_matrix.png')

# 显示混淆矩阵
plt.show()
