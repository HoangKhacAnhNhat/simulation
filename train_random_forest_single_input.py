"""
Giai đoạn 4.3 - Random Forest Regression Single Input
Input: VWC (%); Target: PWP dương (kPa)
Cài đặt: pip install pandas numpy scikit-learn joblib
Chạy:
python train_random_forest_single_input.py --input Giai_doan_4_1_Single_Input_Dataset.csv
Đánh giá Leave-One-Scenario-Out (LOSO), 6 folds.
"""
import argparse, os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

FEATURE = "moisture_vwc_percent"
TARGET = "pore_pressure_positive_kpa"
PARAMS = dict(n_estimators=300, max_depth=None, min_samples_split=2,
              min_samples_leaf=1, max_features=1.0, bootstrap=True,
              random_state=42, n_jobs=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="results_4_3_single_input")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    needed = ["scenario_id", "time_min", FEATURE, TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    if df[needed].isna().any().any():
        raise ValueError("Có giá trị thiếu.")

    pred_parts, metric_rows = [], []
    for scenario in sorted(df.scenario_id.unique()):
        train = df[df.scenario_id != scenario]
        test = df[df.scenario_id == scenario]
        model = RandomForestRegressor(**PARAMS)
        model.fit(train[[FEATURE]], train[TARGET])
        yhat = model.predict(test[[FEATURE]])
        pred_parts.append(pd.DataFrame({
            "scenario_id": scenario, "time_min": test.time_min.to_numpy(),
            "vwc_percent": test[FEATURE].to_numpy(),
            "y_true_kpa": test[TARGET].to_numpy(),
            "y_pred_kpa": yhat, "error_kpa": yhat-test[TARGET].to_numpy()
        }))
        metric_rows.append({
            "test_scenario": scenario, "n_test": len(test),
            "MAE_kPa": mean_absolute_error(test[TARGET], yhat),
            "RMSE_kPa": mean_squared_error(test[TARGET], yhat)**0.5,
            "R2": r2_score(test[TARGET], yhat)
        })
    oof = pd.concat(pred_parts, ignore_index=True)
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(args.output_dir,"4_3_single_input_metrics_by_scenario.csv"),
        index=False,encoding="utf-8-sig")
    oof.to_csv(os.path.join(args.output_dir,"4_3_single_input_LOSO_predictions.csv"),
               index=False,encoding="utf-8-sig")
    pooled = {
        "n":len(oof),
        "MAE_kPa":mean_absolute_error(oof.y_true_kpa,oof.y_pred_kpa),
        "RMSE_kPa":mean_squared_error(oof.y_true_kpa,oof.y_pred_kpa)**0.5,
        "R2":r2_score(oof.y_true_kpa,oof.y_pred_kpa),
        "negative_prediction_count":int((oof.y_pred_kpa<0).sum())
    }
    pd.DataFrame([pooled]).to_csv(
        os.path.join(args.output_dir,"4_3_single_input_overall_metrics.csv"),
        index=False,encoding="utf-8-sig")
    final_model=RandomForestRegressor(**PARAMS).fit(df[[FEATURE]],df[TARGET])
    joblib.dump(final_model,os.path.join(
        args.output_dir,"4_3_single_input_random_forest_full_dataset.joblib"))
    pd.DataFrame({"feature":[FEATURE],
                  "importance":final_model.feature_importances_}).to_csv(
        os.path.join(args.output_dir,"4_3_single_input_feature_importance.csv"),
        index=False,encoding="utf-8-sig")
    print(pd.DataFrame([pooled]).to_string(index=False))
    print(pd.DataFrame(metric_rows).to_string(index=False))

if __name__=="__main__":
    main()
