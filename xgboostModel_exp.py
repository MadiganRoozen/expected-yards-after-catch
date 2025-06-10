from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import shap

def get_category_probabilities(sample_probs, val):
    return {
        "actual yards": val,
        "0 yards": sample_probs[0:0.99].sum(),
        "1-5 yards": sample_probs[1:5.99].sum(),
        "6-10 yards": sample_probs[6:10.99].sum(),
        "11+ yards": sample_probs[11:].sum()
    }

# Data preparation
input_df = pd.read_csv(r"SD/model_input_df_pass_catch_with_friends.csv", 
                      usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", 
                              "receivero", "receiverdir", "distance_to_nearest_def", "defenders_in_path",
                              "friends_in_path", "pass_length", "yards_to_go", "yardline_num", "yards_gained"])

input_df = input_df[input_df["pass_length"] > 0]

# Feature engineering
def bucket_separation(distance):
    if distance < 0.5: return 0
    elif 0.5 <= distance < 2.0: return 1
    elif 2.0 <= distance < 5.0: return 2
    elif 5.0 <= distance < 8.0: return 3
    else: return 4

def bucket_friends(count):
    if count == 0: return 0
    elif 1 <= count <= 2: return 1
    elif 3 <= count <= 4: return 2
    else: return 3

input_df["defender_separation_encoded"] = input_df["distance_to_nearest_def"].apply(bucket_separation)

input_df["friends_bucket"] = input_df["friends_in_path"].apply(bucket_friends)

input_df["blocking_advantage"] = (
    np.log1p(input_df["friends_in_path"]) - 
    np.log1p(input_df["defenders_in_path"])
)
input_df["separation_x_pass_length"] = input_df["defender_separation_encoded"] * input_df["pass_length"]

# 1. Receiver's positioning advantage (sideline vs. middle)
input_df["receiver_near_sideline"] = (np.abs(input_df["receiverx"]) > 30).astype(int)

# 2. Defender density ratio
input_df["defender_density_ratio"] = input_df["defenders_in_path"] / (input_df["distance_to_nearest_def"] + 0.1)

# 3. Directional momentum (receiver speed × direction)
input_df["receiver_momentum"] = input_df["receivers"] * np.cos(np.radians(input_df["receiverdir"]))

# 4. Field position impact (compressed yardline)
input_df["yardline_squeezed"] = np.log1p(input_df["yardline_num"])

# Add receiver's deceleration impact
input_df["receiver_decelerating"] = (input_df["receivera"] < -2).astype(int)

# Suggested replacement:
# Improved version:
input_df["defender_pressure"] = (
    np.sqrt(input_df["defenders_in_path"]) *  # Diminishing returns
    np.where(input_df["receivera"] < 0, 1.5, 0.8) *  # Direction matters
    (1 / np.log1p(input_df["distance_to_nearest_def"])))
      

input_df["defender_pressure"] = np.clip(input_df["defender_pressure"], 0, 5)


input_df["effective_momentum"] = (
    input_df["receivers"] * 
    np.cos(np.radians(input_df["receiverdir"])) *  # Current
    (input_df["yardline_num"] / 100)  # Field position modifier
)

# 3. New: Catch Continuity Feature
input_df["catch_continuity"] = (
    input_df["receiverdis"] / 
    (input_df["pass_length"] + 0.1)) * input_df["receivers"]  # Speed multiplier


# Create tackle probability feature
input_df["tackle_indicators"] = (
    (input_df["defenders_in_path"] >= 2) & 
    (input_df["receivera"] < 0)
).astype(int)


# Prepare features and target
x = input_df[[
    "friends_bucket",
    "defender_separation_encoded",
    "blocking_advantage",
    "separation_x_pass_length",
    "receiverx",
    "receivery",
    "receivers",
    "receivera",
    "receiverdis",
    "receivero",
    "receiverdir",
    "defenders_in_path",
    "pass_length",
    "yards_to_go",
    "yardline_num" ,
    "yardline_squeezed" ,
    "receiver_momentum" ,
    "defender_density_ratio" ,
    "receiver_near_sideline",
    "tackle_indicators",
    "defender_pressure",
    "receiver_decelerating",

]].copy()
 
# Define feature weights (higher = more important)
feature_weights = {
    "defender_pressure": 3.0,  # Most important now
    "effective_momentum": 2.5,
    "catch_continuity": 2.2,
    "defender_separation_encoded": 2.0,
    "friends_bucket": 1.8,
    # Default weight is 1.0 for others
}

# 2. SAMPLE WEIGHTS (for class balancing)
def dynamic_sample_weights(y_true, X):
    """Combines class weights with situational importance"""
    base_weights = {
        0: 2.0,  # 0-yard
        1: 1.3,   # 1-5
        2: 0.7,   # 6-10
        3: 0.5    # 11+
    }
    
    # Field position multiplier (more important near endzone)
    position_boost = 1 + (X["yardline_num"] / 120)
    
    # Defender presence penalty
    defender_penalty = np.where(X["defenders_in_path"] > 2, 1.2, 1.0)
    
    return y_true.map(base_weights) * position_boost * defender_penalty

y = input_df["yards_gained"].clip(lower=0).apply(
    lambda y: 0 if y == 0 else 1 if 1 <= y <= 5 else 2 if 6 <= y <= 10 else 3
)

# Model training
params = {
    "objective": "multi:softprob",
    "num_class": 4,
    "eta": 0.02,  # Reduced learning rate
    "max_depth": 7,  # Slightly deeper
    "subsample": 0.6,  # More aggressive regularization
    "colsample_bytree": 0.5,
    "alpha": 0.5,  # L1 reg
    "lambda": 1.5,  # L2 reg
    "grow_policy": "lossguide",  # Better for imbalanced data
    "max_leaves": 35,  # Alternative to depth control
    "max_delta_step": 1,  # Helps with class imbalance
    
}

kfold = KFold(n_splits=5, shuffle=True, random_state=7)
all_preds = []
all_true = []
all_test_indices = []  # Store original indices
all_models = []  # Store models for each fold

for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
    print(f"Fold {fold + 1}")
    all_test_indices.extend(test_idx)  # Collect original indices
    
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    smote = SMOTE(sampling_strategy={0: 2000, 1: 2000, 2: 1500, 3: 1500},k_neighbors=5)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    
    
    sample_weights = dynamic_sample_weights(y_train, x_train)
    
    dtrain = xgb.DMatrix(
    x_train,
    label=y_train,
    weight=sample_weights,  # Using new dynamic weights
    feature_weights=[feature_weights.get(col, 1.0) for col in x_train.columns]  # New feature weights
    )
    
    dtest = xgb.DMatrix(x_test)
    
    model = xgb.train(params, dtrain, num_boost_round=500)
    preds = model.predict(dtest)
    
    all_preds.append(preds)
    all_true.append(y_test)
    all_models.append(model)  # Store the model for this fold

# Combine results
full_true = pd.concat(all_true)
full_pred_classes = np.argmax(np.vstack(all_preds), axis=1)

# Get misclassified 0-yard plays
misclassified_zero_mask = (full_true == 0) & (full_pred_classes != 0)
misclassified_zero_indices = full_true[misclassified_zero_mask].index
misclassified_zero = x.loc[misclassified_zero_indices].sample(5, random_state=42)
class_names = ["0 yards", "1-5 yards", "6-10 yards", "11+ yards"]

# Create explanation file
with open('incorrect_zero_yards.txt', 'w') as f:
    f.write("INCORRECT 0-YARD PREDICTIONS ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    for i, (play_idx, play) in enumerate(misclassified_zero.iterrows()):
        # [Previous code remains the same until SHAP section]
        
        # 4. SHAP analysis
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(play.to_frame().T)
            
            # Handle multi-output formats
            if isinstance(shap_values, list):
                # For multi-class, use values for predicted class
                shap_values = shap_values[pred_class]
            elif len(shap_values.shape) == 3:
                # For 3D array, take first dimension
                shap_values = shap_values[0]
            
            # Ensure we have matching dimensions
            if len(shap_values) == len(x.columns):
                f.write("\nTOP INFLUENTIAL FEATURES:\n")
                shap_df = pd.DataFrame({
                    'feature': x.columns,
                    'shap_value': shap_values.flatten()[:len(x.columns)]  # Force 1D and trim
                }).sort_values('shap_value', key=abs, ascending=False)
                
                for _, row in shap_df.head(5).iterrows():
                    f.write(f"{row['feature']:>25}: {row['shap_value']:+.4f}\n")
            else:
                f.write("\nSHAP Analysis: Dimension mismatch (expected {len(x.columns)} features, got {len(shap_values)} values)\n")

        except Exception as e:
            f.write(f"\nSHAP Analysis Failed: {str(e)}\n")
            f.write("Model may need retraining or different explainer.\n")
        
        f.write("\n" + "="*60 + "\n")


        # Get detailed false negatives (actual 0-yard plays predicted as >0)
false_negatives = x.loc[(y == 0) & (full_pred_classes != 0)]
false_negatives = x.loc[(y == 0) & (full_pred_classes != 0)].copy()

# Analyze their characteristics using the engineered features
print("\nCommon patterns in false negatives:")
print(false_negatives[['defenders_in_path', 'defender_separation_encoded', 'friends_bucket']].value_counts().head(10))

# If you want the original values (from input_df), merge them back:
false_negatives_with_original = false_negatives.merge(
    input_df[['distance_to_nearest_def', 'friends_in_path']],
    left_index=True,
    right_index=True,
    how='left'
)
# Analyze their characteristics
print(false_negatives.describe())


# Plot feature distributions for false negatives vs correct 0-yard predictions
plt.figure(figsize=(12,6))
sns.kdeplot(data=input_df.loc[(y == 0) & (full_pred_classes == 0), 'defender_pressure'], 
            label='Correct 0-yard')
sns.kdeplot(data=false_negatives['defender_pressure'], 
            label='False Negative')
plt.title('Defender Pressure Distribution Comparison')
plt.legend()

# Combine results
preds_concat = np.vstack(all_preds)
true_labels = pd.concat(all_true, ignore_index=True)
predicted_classes = np.argmax(preds_concat, axis=1)

# Evaluation
class_names = ["0 yards", "1-5 yards", "6-10 yards", "11+ yards"]
cm = confusion_matrix(true_labels, predicted_classes)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_labels, predicted_classes, target_names=class_names))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.show()

print("Analysis complete. Saved to 'incorrect_zero_yards.txt'")



