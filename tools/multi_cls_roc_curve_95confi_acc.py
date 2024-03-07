import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


dpi = 300

# Function to parse strings from the DataFrame columns
def parse_tensor(tensor_str):
    tensor_str = tensor_str.replace('tensor', '').replace('(', '').replace(')', '')
    return eval(tensor_str)

# Function to calculate the confidence interval for AUC
def compute_auc_confidence_interval(auc_score, n1, n2, alpha=0.95):
    q1 = norm.ppf((1 + alpha) / 2)
    q2 = (2 * n1 * n2 + auc_score**2 * (n1 + n2 - 1)) / (n1 * n2)
    se_auc = np.sqrt(auc_score * (1 - auc_score) + (auc_score - 1) / (n1 - 1) + (auc_score - 1) / (n2 - 1)) / np.sqrt(n2)
    auc_confidence_lower = auc_score - q1 * se_auc
    auc_confidence_upper = auc_score + q1 * se_auc
    return auc_confidence_lower, auc_confidence_upper

# Read the CSV file into a DataFrame
df = pd.read_csv('result_.csv')

# Apply the parsing function to each column
df['pred_score'] = df['pred_score'].apply(parse_tensor)
df['pred_label'] = df['pred_label'].apply(lambda x: parse_tensor(x)[0])
df['gt_label'] = df['gt_label'].apply(lambda x: parse_tensor(x)[0])

# Binarize the output labels for ROC computation
y_true = label_binarize(df['gt_label'], classes=np.unique(df['gt_label']))
y_scores = np.vstack(df['pred_score'])

# Compute ROC curve and ROC AUC for each class
n_classes = y_true.shape[1]
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    auc_score = roc_auc[i]
    n1 = sum(y_true[:, i])
    n2 = len(y_true[:, i]) - n1
    lower, upper = compute_auc_confidence_interval(auc_score, n1, n2)
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {i} (area = {auc_score:0.2f}, 95% CI: [{lower:0.2f}-{upper:0.2f}])')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC with 95% Confidence Intervals')
plt.legend(loc="lower right")
plt.savefig('roc_curves_with_ci.png', dpi=dpi)
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(df['gt_label'], df['pred_label'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=dpi)
plt.show()

def calculate_performance_metrics(y_true, y_pred, class_label):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.sum() - cm[class_label, :].sum() - cm[:, class_label].sum() + cm[class_label, class_label]
    fp = cm[:, class_label].sum() - cm[class_label, class_label]
    fn = cm[class_label, :].sum() - cm[class_label, class_label]
    tp = cm[class_label, class_label]

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    return sensitivity, specificity, precision


performance_metrics = pd.DataFrame(columns=['Class', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall'])

for i, class_label in enumerate(np.unique(df['gt_label'])):
    y_pred_class = df['pred_label'] == class_label
    y_true_class = df['gt_label'] == class_label

    accuracy = accuracy_score(y_true_class, y_pred_class)
    sensitivity, specificity, precision = calculate_performance_metrics(df['gt_label'], df['pred_label'], class_label)
    recall = sensitivity

    performance_metrics.loc[i] = [class_label, accuracy, sensitivity, specificity, precision, recall]

performance_metrics.to_csv('performance_metrics.csv', index=False)
