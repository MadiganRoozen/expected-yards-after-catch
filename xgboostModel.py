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
input_df = pd.read_csv(r"model_input_df_pass_catch_with_friends.csv", 
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
input_df["blocking_advantage"] = input_df["friends_bucket"] - np.floor(input_df["defenders_in_path"] / 2)
input_df["separation_x_pass_length"] = input_df["defender_separation_encoded"] * input_df["pass_length"]

# 1. Receiver's positioning advantage (sideline vs. middle)
input_df["receiver_near_sideline"] = (np.abs(input_df["receiverx"]) > 30).astype(int)

# 2. Defender density ratio
input_df["defender_density_ratio"] = input_df["defenders_in_path"] / (input_df["distance_to_nearest_def"] + 0.1)

# 3. Directional momentum (receiver speed × direction)
input_df["receiver_momentum"] = input_df["receivers"] * np.cos(np.radians(input_df["receiverdir"]))

# 4. Field position impact (compressed yardline)
input_df["yardline_squeezed"] = np.log1p(input_df["yardline_num"])


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
]].copy()
 
# Define feature weights (higher = more important)
feature_weights = {
    "friends_bucket": 1.5,               # 3x weight - most critical for YAC
    "blocking_advantage": 1.5,            # 2.5x weight
    "defender_density_ratio": 2.75,        # 2x weight
    "defender_separation_encoded": 2.0,   # 1.8x weight
    "separation_x_pass_length": 1.5,      # 1.5x weight
    "receiver_momentum": 1.5,             # New dynamic feature
    "receivers": 3.0,
    "receiverdir": 2.5,
    "pass_length": 2.25,
    "receivery": 2.0,
    # All other features get default weight of 1.0
}

y = input_df["yards_gained"].clip(lower=0).apply(
    lambda y: 0 if y == 0 else 1 if 1 <= y <= 5 else 2 if 6 <= y <= 10 else 3
)
print(pd.Series(y).value_counts().sort_index())

# Model training
params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 4,
    "eta": 0.05,
    "gamma": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "max_depth": 5,
    "min_child_weight": 1,
}

kfold = KFold(n_splits=5, shuffle=True, random_state=7)
all_preds = []
all_true = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
    print(f"Fold {fold + 1}")
    
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    smote = SMOTE(sampling_strategy={2: 1300, 3: 1100}, k_neighbors=4, random_state=7)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    weight_dict[1] = 0.8  # Adjust weight for 1-5 yards class
    
    sample_weights = y_train.map(weight_dict)
    
    # Apply feature weights here
    dtrain = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=sample_weights,
        feature_weights=[feature_weights.get(col, 1.0) for col in x_train.columns]
    )
    
    dtest = xgb.DMatrix(x_test)
    
    model = xgb.train(params, dtrain, num_boost_round=500)
    preds = model.predict(dtest)
    all_preds.append(preds)
    all_true.append(y_test.reset_index(drop=True))

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


# Select 5 random test plays
sample_plays = x_test.sample(5, random_state=42)

# Create explanation file
with open('test.txt', 'w') as f:
    f.write("XGBOOST DECISION TREE ANALYSIS\n")
    f.write("="*50 + "\n\n")
    
    for i, (_, play) in enumerate(sample_plays.iterrows()):
        f.write(f"\nPLAY #{i+1} - ACTUAL YARDS: {y_test.loc[play.name]}\n")
        f.write("-"*50 + "\n")
        
        # 1. Write all play features
        f.write("\nPLAY DATA:\n")
        for feat, val in play.items():
            f.write(f"{feat:>25}: {val:.3f}\n")
        
        # 2. Get tree decisions
        f.write("\nDECISION PATH:\n")
        leaf_preds = model.predict(xgb.DMatrix(play.to_frame().T), pred_leaf=True)
        
        # For each tree in the ensemble
        for tree_idx in range(5):  # First 5 trees
            f.write(f"\nTree {tree_idx}:\n")
            tree = model.get_dump()[tree_idx]
            
            # Find the leaf for this play
            leaf_id = leaf_preds[0][tree_idx]
            
            # Extract the decision path
            for line in tree.split('\n'):
                if f'leaf={leaf_id}' in line:
                    f.write(line + "\n")
                    break
                elif '[' in line:  # Decision node
                    f.write(line + "\n")
        
        # 3. Final prediction
        pred = model.predict(xgb.DMatrix(play.to_frame().T))
        f.write(f"\nFINAL PREDICTION: {class_names[np.argmax(pred)]} ({np.max(pred):.1%} confidence)\n")
        f.write("="*50 + "\n")

        '''
        # 1. Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)

        # 2. Plot summary of feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_test, plot_type="bar", class_names=class_names)
        plt.title("SHAP Feature Importance (Absolute Impact)")
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png')  # Save for later reference
        plt.show()

        # 3. Visualize decision logic for specific plays
        for i in range(min(3, len(sample_plays))):  # First 3 sample plays
            plt.figure(figsize=(12, 5))
            shap.decision_plot(
                explainer.expected_value,
                shap_values[i],
                x_test.iloc[i],
                feature_names=list(x_test.columns),
                show=False
            )
            plt.title(f"SHAP Decision Plot - Play {i+1} (Actual: {y_test.iloc[i]} yards)")
            plt.tight_layout()
            plt.savefig(f'shap_play_{i+1}.png')
            plt.close()  # Close plot to prevent display if running in notebook
            '''
model.save_model("xgbModel.json")

loaded_model = xgb.Booster()
loaded_model.load_model("xgbModel.json")

input_df = pd.read_csv(r"ranking_test_subset.csv", 
                      usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", 
                              "receivero", "receiverdir", "distance_to_nearest_def", "defenders_in_path",
                              "friends_in_path", "pass_length", "yards_to_go", "yardline_num"])

input_df = input_df[input_df["pass_length"] > 0]

pass_lengths = input_df["pass_length"]

input_df["defender_separation_encoded"] = input_df["distance_to_nearest_def"].apply(bucket_separation)
input_df["friends_bucket"] = input_df["friends_in_path"].apply(bucket_friends)
input_df["blocking_advantage"] = input_df["friends_bucket"] - np.floor(input_df["defenders_in_path"] / 2)
input_df["separation_x_pass_length"] = input_df["defender_separation_encoded"] * input_df["pass_length"]

# 1. Receiver's positioning advantage (sideline vs. middle)
input_df["receiver_near_sideline"] = (np.abs(input_df["receiverx"]) > 30).astype(int)

# 2. Defender density ratio
input_df["defender_density_ratio"] = input_df["defenders_in_path"] / (input_df["distance_to_nearest_def"] + 0.1)

# 3. Directional momentum (receiver speed × direction)
input_df["receiver_momentum"] = input_df["receivers"] * np.cos(np.radians(input_df["receiverdir"]))

# 4. Field position impact (compressed yardline)
input_df["yardline_squeezed"] = np.log1p(input_df["yardline_num"])

# Prepare features and target
ranking_test = input_df[[
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
]].copy()
 
ranking_test = xgb.DMatrix(ranking_test)

# You can now use it for prediction
ranking_pred = loaded_model.predict(ranking_test)

print(ranking_pred)
predicted_classes = np.argmax(ranking_pred, axis=1)
print(predicted_classes)

results = pd.DataFrame({
    'predicted_class': predicted_classes,
    'pass_length': pass_lengths
})

print(results)

#Deciding rankings
sample_max_throw_dist = 12

#results now only includes passes that are within the players range
results = results[results["pass_length"] <= sample_max_throw_dist]

rankings = results.sort_values(by=['predicted_class', 'pass_length'], ascending=[False, True])

# Add a rank column starting at 1
rankings["rank"] = range(1, len(rankings) + 1)

print(rankings)

