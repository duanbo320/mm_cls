from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./result.csv')

# Initialize a list to store the bootstrap AUC scores
bootstrap_auc_scores = []

# Number of bootstrap samples to create
n_bootstrap_samples = 1000

# Define the true labels and the predicted probabilities
true_labels = data['gt_label']
predicted_probabilities = data['pred_score_normal']

# Calculate the AUC score
auc_score = roc_auc_score(true_labels, predicted_probabilities)
print(f'Original AUC score: {auc_score:.3f}')

# Generate bootstrap samples and calculate AUC scores
for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample
    bootstrap_sample = resample(data, replace=True, n_samples=len(data), random_state=None)

    # Define the true labels and the predicted probabilities for the bootstrap sample
    bootstrap_true_labels = bootstrap_sample['gt_label']
    bootstrap_predicted_probabilities = bootstrap_sample['pred_score_normal']

    # Calculate the AUC score for the bootstrap sample and add it to the list
    bootstrap_auc_score = roc_auc_score(bootstrap_true_labels, bootstrap_predicted_probabilities)
    bootstrap_auc_scores.append(bootstrap_auc_score)

# Calculate the 95% confidence interval for the AUC score
confidence_lower = np.percentile(bootstrap_auc_scores, 2.5)
confidence_upper = np.percentile(bootstrap_auc_scores, 97.5)
print(f'95% Confidence interval for the AUC score: ({confidence_lower:.3f}, {confidence_upper:.3f})')



# Set the DPI and figsize (width, height) in inches
dpi = 300
figsize = (10, 6)



# Create a new figure with the specified DPI and size
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

# Create a histogram of the bootstrap AUC scores
ax.hist(bootstrap_auc_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Add a vertical line for the original AUC score
ax.axvline(auc_score, color='red', linestyle='dashed', linewidth=2, label=f'Original AUC = {auc_score:.3f}')

# Add vertical lines for the 95% confidence interval
ax.axvline(confidence_lower, color='green', linestyle='dashed', linewidth=2, label=f'2.5 percentile = {confidence_lower:.3f}')
ax.axvline(confidence_upper, color='blue', linestyle='dashed', linewidth=2, label=f'97.5 percentile = {confidence_upper:.3f}')

# Add labels and a legend
ax.set_title('Bootstrap AUC Scores with 95% Confidence Interval')
ax.set_xlabel('AUC Score')
ax.set_ylabel('Frequency')
ax.legend()

# Save the figure to a file
plt.savefig('./bootstrap_auc_scores.png', dpi=dpi)

# Show the plot
plt.show()

# Calculate the ROC curve for the original data
fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)

# Calculate the AUC for the original ROC curve
original_auc = auc(fpr, tpr)

# Initialize a list to store the bootstrap AUC scores for the ROC curve
bootstrap_auc_scores_roc = []

# Generate bootstrap samples and calculate AUC scores for the ROC curve
for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample
    bootstrap_sample = resample(data, replace=True, n_samples=len(data), random_state=None)

    # Define the true labels and the predicted probabilities for the bootstrap sample
    bootstrap_true_labels = bootstrap_sample['gt_label']
    bootstrap_predicted_probabilities = bootstrap_sample['pred_score_ill']

    # Calculate the ROC curve for the bootstrap sample
    bootstrap_fpr, bootstrap_tpr, _ = roc_curve(bootstrap_true_labels, bootstrap_predicted_probabilities)

    # Calculate the AUC for the bootstrap ROC curve and add it to the list
    bootstrap_auc = auc(bootstrap_fpr, bootstrap_tpr)
    bootstrap_auc_scores_roc.append(bootstrap_auc)

# Calculate the 95% confidence interval for the AUC of the ROC curve
confidence_lower_roc = np.percentile(bootstrap_auc_scores_roc, 2.5)
confidence_upper_roc = np.percentile(bootstrap_auc_scores_roc, 97.5)

# Print the original AUC and the 95% confidence interval
print(f'Original AUC for the ROC curve: {original_auc:.3f}')
print(f'95% Confidence interval for the AUC of the ROC curve: ({confidence_lower_roc:.3f}, {confidence_upper_roc:.3f})')

# Plot
plt.figure(figsize=figsize, dpi=dpi)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve\nAUC = {original_auc:.3f}\n95% CI = ({confidence_lower_roc:.3f}, {confidence_upper_roc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic with 95% Confidence Interval')
plt.legend(loc="lower right")

# Save the figure
plt.savefig('./roc_curve_with_ci.png', dpi=dpi)

# Show the plot
plt.show()
