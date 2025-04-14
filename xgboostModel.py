from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm 
import matplotlib.pyplot as plt


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
input_df = pd.read_csv("model_input_df_2.csv", usecols=["receiverx", "receivery", "receivers", "receivera", "receiverdis", "receivero", "receiverdir", "distance_to_nearest_def", "defenderx",
     "defendery", "defenders", "defendera", "defenderdis", "defendero", "defenderdir",
    "defenders_in_path","pass_length", "yards_to_go", "yardline_num", "yards_gained"])

#input_df = input_df[input_df["yards_gained"]<=22]
input_df = input_df[input_df["yards_gained"]>=0]
x = input_df.drop(columns=["yards_gained"])
y = input_df["yards_gained"]




kfold = KFold(n_splits=5, shuffle=True, random_state=7)
all_preds = []
all_true = []

params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 76,
    "eta": 0.025,
    "gamma": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 4,
    "min_child_weight": 1,
}

for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
    print(f"Fold {fold + 1}")

    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    model = xgb.train(params, dtrain, num_boost_round=500)

    preds = model.predict(dtest)
    all_preds.append(preds)
    all_true.append(y_test.reset_index(drop=True))

#combine all predictions and labels
preds_concat = np.vstack(all_preds)
true_labels = pd.concat(all_true, ignore_index=True)

predicted_classes = np.argmax(preds_concat, axis=1)

# Now calculate category probabilities for each sample
category_probs = [
    get_category_probabilities(sample_probs, true_labels.iloc[i]) 
    for i, sample_probs in enumerate(preds_concat)
]

# Example: Print category probabilities for the first sample
print("Category probabilities for the first sample:")
print(category_probs[0])
print(category_probs[1])
print(category_probs[2])
print(category_probs[3])
print(category_probs[4])

#dataFrame for actual vs predicted values
results_df = pd.DataFrame({
    'Actual Value (Yards Gained)': true_labels,
    'Predicted Value (Class)': predicted_classes,
    'Predicted Probability (Max Class Prob)': np.max(preds_concat, axis=1) 
})

print(results_df.head())






def map_outcome(label):
    if label == 0:
        return "loss"
    elif 1 <= label <= 5:
        return "short_gain"
    elif 6 <= label <= 10:
        return "med_gain"
    else:
        return "long_gain"

true_outcomes = true_labels.apply(map_outcome)

loss_prob = preds_concat[:, 0]        
short_gain_prob = preds_concat[:, 1:6].sum(axis=1) 
med_gain_prob = preds_concat[:, 6:10].sum(axis=1)   
long_gain_prob = preds_concat[:, 11:].sum(axis=1)    

distance_bin = x['pass_length'].reset_index(drop=True).apply(lambda val: "1: Short" if val < 5 else "2: Long")

df = pd.DataFrame({
    "outcome": true_outcomes,
    "distance": distance_bin,
    "loss": loss_prob,
    "short_gain": short_gain_prob,
    "med_gain": med_gain_prob,
    "long_gain": long_gain_prob
})

df_long = df.melt(id_vars=["outcome", "distance"], 
                  value_vars=["loss", "short_gain", "med_gain", "long_gain"],
                  var_name="type", value_name="pred_prob")


df_long["bin_pred_prob"] = (df_long["pred_prob"] / 0.05).round() * 0.05

# Calculate correctness
df_long["correct"] = (df_long["outcome"] == df_long["type"]).astype(int)

# Group by bin, type, and distance
calibration_df = df_long.groupby(["type", "distance", "bin_pred_prob"]).agg(
    n_plays=("correct", "count"),
    n_correct=("correct", "sum")
).reset_index()

calibration_df["bin_actual_prob"] = calibration_df["n_correct"] / calibration_df["n_plays"]

# Rename labels to match R version
calibration_df["type"] = calibration_df["type"].replace({
    "loss": "0 yards",
    "short_gain": "1-5 yards",
    "med_gain": "6-10 yards",
    "long_gain": "11+ yards"
})

# Order types for plotting
type_order = ["0 yards", "1-5 yards", "6-10 yards", "11+ yards"]
calibration_df["type"] = pd.Categorical(calibration_df["type"], categories=type_order, ordered=True)

sns.set(style="whitegrid")
g = sns.FacetGrid(
    calibration_df[calibration_df["n_plays"] > 15],
    col="type",
    row="distance",
    margin_titles=True,
    height=4,
    aspect=1
)

g.map_dataframe(
    sns.scatterplot,
    x="bin_pred_prob",
    y="bin_actual_prob",
    size="n_plays",
    sizes=(20, 200),
    legend=False,
    color="steelblue"
)

g.map_dataframe(
    sns.regplot,
    x="bin_pred_prob",
    y="bin_actual_prob",
    scatter=False,
    lowess=True,
    color="darkorange"
)

#x=y line to show the perfect match
for ax in g.axes.flatten():
    ax.plot([0, 1], [0, 1], ls="--", color="black")

g.set(xlim=(0, 1), ylim=(0, 1))
g.set_axis_labels("Estimated yards after catch", "Observed yards after catch")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Calibration by Gain Type and Air Yards")

plt.show()



#calculating calibration error 


#compute the absolute difference between predicted and actual binned probabilities
calibration_df["cal_diff"] = (calibration_df["bin_pred_prob"] - calibration_df["bin_actual_prob"]).abs()

#group by `distance` and compute weighted mean calibration error
cal_error_df = (
    calibration_df
    .groupby("distance")
    .apply(lambda g: pd.Series({
        "weight_cal_error": np.average(g["cal_diff"], weights=g["n_plays"]),
        "n_correct": g["n_correct"].sum()
    }))
    .reset_index()
)

# Step 3: Compute the overall weighted average calibration error
overall_cal_error = np.average(cal_error_df["weight_cal_error"], weights=cal_error_df["n_correct"])

# Optional: Round it for display
print("Weighted Calibration Error:", round(overall_cal_error, 4))
