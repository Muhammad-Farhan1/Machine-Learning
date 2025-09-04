# Importing evaluation metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------
# Ground Truth (Actual values)
# 0 = Not Spam, 1 = Spam
y_true = [0, 0, 1, 1, 0, 1, 0, 1]

# Predictions made by our model
y_pred = [0, 0, 0, 0, 0, 1, 1, 1]
# ---------------------------------------

# Accuracy: (Correct Predictions / Total Predictions)
# Tells us how many predictions our model got right overall.
accuracy = accuracy_score(y_true, y_pred)

# Precision: (True Positives / (True Positives + False Positives))
# Out of all the emails predicted as "Spam", how many were actually Spam?
precision = precision_score(y_true, y_pred)

# Recall: (True Positives / (True Positives + False Negatives))
# Out of all the actual "Spam" emails, how many did the model correctly identify?
recall = recall_score(y_true, y_pred)

# F1 Score: (2 * Precision * Recall) / (Precision + Recall)
# The harmonic mean of Precision and Recall.
# Good when we want a balance between Precision and Recall.
f1 = f1_score(y_true, y_pred)

# ---------------------------------------
# Printing results in a clean format
print("📊 Model Evaluation Results")
print("-" * 35)
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")
print("-" * 35)

# Extra Note:
# - High Precision means fewer false alarms (important when marking spam).
# - High Recall means catching most spam emails (important to not miss spam).
# - F1 Score balances both (useful when dataset is imbalanced).
