import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for better visualization

dpi= 300

# Function to parse strings from the DataFrame columns
def parse_tensor(tensor_str):
    # Remove the 'tensor' keyword and parse the remaining string as a list or a single integer
    tensor_str = tensor_str.replace('tensor', '').replace('(', '').replace(')', '')
    return eval(tensor_str)

# Read the CSV file into a DataFrame
df = pd.read_csv('result_.csv')

# Apply the parsing function to each column
df['pred_score'] = df['pred_score'].apply(parse_tensor)
df['pred_label'] = df['pred_label'].apply(lambda x: parse_tensor(x)[0])  # Extract single values
df['gt_label'] = df['gt_label'].apply(lambda x: parse_tensor(x)[0])  # Extract single values

# Binarize the output labels for ROC computation
y_true = label_binarize(df['gt_label'], classes=np.unique(df['gt_label']))
y_scores = np.vstack(df['pred_score'])

# Compute ROC curve and ROC AUC for each class
n_classes = y_true.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC')
plt.legend(loc="lower right")

# Save the ROC curve plot
plt.savefig('roc_curves.png', dpi=300)
plt.show()


# Calculate confusion matrix
cm = confusion_matrix(df['gt_label'], df['pred_label'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png',dpi=dpi)
plt.show()
