from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


def get_category_probabilities(sample_probs, val):
    actual_yards = val
    # Categories: 0 yards, 1-5 yards, 6-10 yards, 11+ yards
    loss_prob = sample_probs[0]  # Class 0: 0 yards
    short_gain_prob = sample_probs[1:6].sum()  # Classes 1-5: 1-5 yards
    med_gain_prob = sample_probs[6:11].sum()  # Classes 6-10: 6-10 yards
    long_gain_prob = sample_probs[11:].sum()  # Classes 11-76: 11+ yards

    return {
        "actual yards": actual_yards,
        "0 yards": loss_prob,
        "1-5 yards": short_gain_prob,
        "6-10 yards": med_gain_prob,
        "11+ yards": long_gain_prob
    }


#Data preparation
input_df = pd.read_csv("model_input_df.csv", usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", 
    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"])

#input_df = pd.read_csv("model_input_df_2.csv", usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", "defenderx",
#    "defendery", "defenders", "defendera", "defenderdis", "defendero", "defenderdir",
#    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"])

#input_df = input_df[input_df["yards_gained"]<=30]
#input_df = input_df[input_df["yards_gained"]>=0]
x = input_df.drop(columns=["yards_gained"])
y = input_df["yards_gained"]
y = input_df["yards_gained"].clip(lower=0)

def bucketize(y):
    if y == 0:
        return 0
    elif 1 <= y <= 5:
        return 1
    elif 6 <= y <= 10:
        return 2
    else:
        return 3

y = input_df["yards_gained"].apply(bucketize)

kfold = KFold(n_splits=5, shuffle=True, random_state=7)
all_preds = []
all_true = []

params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 4,
    "eta": 0.05,
    "gamma": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 5,
    "min_child_weight": 1,
}

for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
    print(f"Fold {fold + 1}")

    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    #smote = SMOTE(sampling_strategy='auto', random_state=7, k_neighbors=2)
    #x_train, y_train = smote.fit_resample(x_train, y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = dict(zip(np.unique(y_train), class_weights))

    class_to_adjust = 1  # Specify the class you want to modify
    new_weight = 0.8      # Specify the new weight you want to assign to this class

    # Modify the weight for the specified class
    weight_dict[class_to_adjust] = new_weight

    #weight_dict = {
    #0: 1.14,  
    #1: 1.2,  
    #2: 2.3,  
    #3: 2.7
    #}

    print("Class weight dictionary:", weight_dict)

    # Create sample weights for each row in training data
    sample_weights = y_train.map(weight_dict)

    # Build DMatrix with sample weights
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(x_test)

    model = xgb.train(params, dtrain, num_boost_round=500)

    importances = model.get_score(importance_type='gain')  # or 'weight', 'cover', 'total_gain'

    # Convert to DataFrame for easy viewing
    importance_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)


    preds = model.predict(dtest)
    all_preds.append(preds)
    all_true.append(y_test.reset_index(drop=True))

#combine all predictions and labels
preds_concat = np.vstack(all_preds)
true_labels = pd.concat(all_true, ignore_index=True)

predicted_classes = np.argmax(preds_concat, axis=1)

# `true_labels` and `predicted_classes` are already defined
class_names = ["0 yards", "1-5", "6-10", "11+"]

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Print raw confusion matrix
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_classes, target_names=class_names))

# Optional: plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()
# Now calculate category probabilities for each sample
#category_probs = [
#    get_category_probabilities(sample_probs, true_labels.iloc[i]) 
#    for i, sample_probs in enumerate(preds_concat)
#]

# Example: Print category probabilities for the first sample
#print("Category probabilities for the first sample:")
#print(category_probs[0])
