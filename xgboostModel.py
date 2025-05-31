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
import os

def get_category_probabilities(sample_probs, val):
    return {
        "actual yards": val,
        "0 yards": sample_probs[0:0.99].sum(),
        "1-5 yards": sample_probs[1:5.99].sum(),
        "6-10 yards": sample_probs[6:10.99].sum(),
        "11+ yards": sample_probs[11:].sum()
    }

def model_train_and_save(training_csv1, training_csv2):
    # Data preparation
    df1 = pd.read_csv(training_csv1, 
                        usecols=["receiverx","receivery","receivers","receivera","receiverdis","receivero","receiverdir","distance_to_nearest_def","defenderx",
                        "defendery","defenders","defendera","defenderdis","defendero","defenderdir","defenders_in_path","friends_in_path","pass_length",
                        "yards_to_go","yardline_num","yards_gained"])
    df2 = pd.read_csv(training_csv2, 
                        usecols=["receiverx","receivery","receivers","receivera","receiverdis","receivero","receiverdir","distance_to_nearest_def","defenderx",
                        "defendery","defenders","defendera","defenderdis","defendero","defenderdir","defenders_in_path","friends_in_path","pass_length",
                        "yards_to_go","yardline_num","yards_gained"])

    input_df = pd.concat([df1, df2], ignore_index=True)

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


    print(input_df.shape)
    input_df = input_df[input_df["pass_length"] > 0]
    input_df["yards_gained"] = input_df['yards_gained'].round()
    print(input_df.shape)
    input_df["defender_separation_encoded"] = input_df["distance_to_nearest_def"].apply(bucket_separation)
    input_df["friends_bucket"] = input_df["friends_in_path"].apply(bucket_friends)
    input_df["blocking_advantage"] = input_df["friends_bucket"] - np.floor(input_df["defenders_in_path"] / 2)
    #input_df['is_screen_pass'] = (input_df['pass_length'] < 0).astype(int)
    #input_df["high_quality_screen"] = ((input_df['is_screen_pass'] == 1) & 
    #                                (input_df["friends_bucket"] >= 2) & 
    #                                (input_df["defender_separation_encoded"] >= 3)).astype(int)
    input_df["separation_x_pass_length"] = input_df["defender_separation_encoded"] * input_df["pass_length"]
    #input_df['screen_yac'] = input_df['yards_gained'] - input_df['pass_length']

    # Prepare features and target
    x = input_df[[
        "friends_bucket",
        "defender_separation_encoded",
        "blocking_advantage",
        #"high_quality_screen",
        "separation_x_pass_length",
        #"screen_yac",
        "receiverx",
        "receivery",
        "receivers",
        "receivera",
        "receiverdis",
        "receivero",
        "receiverdir",
        "defenderx",
        "defendery",
        "defenders",
        "defendera",
        "defenderdis",
        "defendero",
        "defenderdir",
        "defenders_in_path",
        "pass_length",
        "yards_to_go",
        "yardline_num"
    ]].copy()

    y = input_df["yards_gained"].clip(lower=0).apply(
        lambda y: 0 if y == 0 else 1 if 1 <= y <= 5 else 2 if 6 <= y <= 10 else 3
    )

    # Model training
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
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
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


    """
    # Save misclassified 11+ yard plays
    analysis_df = x_test.copy()
    analysis_df['true_yards'] = y_test
    analysis_df['predicted_class'] = predicted_classes
    analysis_df['prob_11plus'] = preds_concat[:, 3]

    misclassified_11plus = analysis_df[(analysis_df['true_yards'] == 3) & (analysis_df['predicted_class'] != 3)]
    with open('11plus_results.txt', 'w') as f:
        f.write(f"Misclassified 11+ Yards Plays: {len(misclassified_11plus)}/{len(y_test[y_test==3])}\n")
        for _, row in misclassified_11plus.nlargest(10, 'prob_11plus').iterrows():
            f.write(f"\nTrue: 11+ | Predicted: {class_names[row['predicted_class']]}\n")
            f.write(f"Prob(11+): {row['prob_11plus']:.2f}\n")
            f.write(f"Blockers: {row['friends_bucket']} | Defenders: {row['defenders_in_path']}\n")
            f.write(f"Pass Length: {row['pass_length']:.1f} | Separation: {row['defender_separation_encoded']}\n")


    # First get ALL correctly predicted 11+ yard plays
    correct_11plus = analysis_df[
        (analysis_df['true_yards'] == 3) & 
        (analysis_df['predicted_class'] == 3)
    ]

    # Then take a sample (15 or fewer if less available)
    sample_size = min(15, len(correct_11plus))
    sampled_plays = correct_11plus.sample(sample_size)

    # Write to file
    with open('correct_11plus_plays.txt', 'w') as f:
        f.write("Correctly Predicted 11+ Yard Plays Analysis\n")
        f.write("==========================================\n\n")
        f.write(f"Total Correct: {len(correct_11plus)}/{len(y_test[y_test==3])} ({len(correct_11plus)/len(y_test[y_test==3]):.1%})\n\n")
        
        for idx, row in sampled_plays.iterrows():
            f.write(f"Play #{idx}\n")
            f.write("------------\n")
            f.write(f"Prob(11+): {row['prob_11plus']:.2f}\n")
            f.write(f"Blockers: {row['friends_bucket']} | Defenders: {row['defenders_in_path']}\n")
            f.write(f"Pass Length: {row['pass_length']:.1f} yds\n")
            f.write(f"Defender Separation: {row['defender_separation_encoded']}\n")
            f.write(f"Blocking Advantage: {row['blocking_advantage']:.1f}\n")
            f.write(f"High Quality Screen: {'Yes' if row['high_quality_screen'] else 'No'}\n")
            f.write(f"YAC Estimate: {row['screen_yac']:.1f} yds\n\n")
    """
    model.save_model("xgbModel.json")
    print("Saved to:", os.path.abspath("xgbModel.json"))

def use_model(testing_csv):

    model = xgb.Booster()
    model.load_model("xgbModel.json")

    input_df = pd.read_csv(testing_csv, 
                        usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", 
                                "receivero", "receiverdir", "distance_to_nearest_def", "defenders_in_path",
                                "friends_in_path", "pass_length", "yards_to_go", "yardline_num"])

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
    input_df['is_screen_pass'] = (input_df['pass_length'] < 0).astype(int)
    input_df["high_quality_screen"] = ((input_df['is_screen_pass'] == 1) & 
                                    (input_df["friends_bucket"] >= 2) & 
                                    (input_df["defender_separation_encoded"] >= 3)).astype(int)
    input_df["separation_x_pass_length"] = input_df["defender_separation_encoded"] * input_df["pass_length"]

    input_data = input_df[[
        "friends_bucket",
        "defender_separation_encoded",
        "blocking_advantage",
        "high_quality_screen",
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
        "yardline_num"
    ]].copy()
    input_data = xgb.DMatrix(input_data)
    sample_classifications = model.predict(input_data)

    print(sample_classifications)



model_train_and_save("model_input_df_pass_release_with_friends.csv", "2023_training.csv")
#use_model("ranking_test_subset.csv")

