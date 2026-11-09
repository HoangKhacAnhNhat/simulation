"""
Giai doan 4.4 - XGBoost Single Input
Input: moisture_vwc_percent (% VWC)
Target: pore_pressure_positive_kpa (kPa)
Danh gia Leave-One-Scenario-Out (LOSO), khong tuning.
Cai dat: pip install pandas scikit-learn xgboost
Chay:
python train_xgboost_single_input.py --input Giai_doan_4_1_Single_Input_Dataset.csv
"""
import argparse, os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE="moisture_vwc_percent"
TARGET="pore_pressure_positive_kpa"
PARAMS=dict(n_estimators=300,max_depth=4,learning_rate=0.05,
            subsample=0.8,colsample_bytree=1.0,
            objective="reg:squarederror",random_state=42,
            n_jobs=-1,reg_lambda=1.0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output_dir",default="results_4_4_single_input")
    a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True)
    df=pd.read_csv(a.input)
    needed=["scenario_id","time_min",FEATURE,TARGET]
    missing=[c for c in needed if c not in df.columns]
    if missing: raise ValueError(f"Thieu cot: {missing}")
    if df[needed].isna().any().any(): raise ValueError("Co gia tri thieu.")
    preds=[]; rows=[]
    for scenario in sorted(df.scenario_id.unique()):
        tr=df[df.scenario_id!=scenario]; te=df[df.scenario_id==scenario]
        model=XGBRegressor(**PARAMS)
        model.fit(tr[[FEATURE]],tr[TARGET],verbose=False)
        yh=model.predict(te[[FEATURE]])
        preds.append(pd.DataFrame({
            "scenario_id":scenario,"time_min":te.time_min.to_numpy(),
            "vwc_percent":te[FEATURE].to_numpy(),
            "y_true_kpa":te[TARGET].to_numpy(),"y_pred_kpa":yh,
            "error_kpa":yh-te[TARGET].to_numpy()}))
        rows.append({"test_scenario":scenario,"n_test":len(te),
            "MAE_kPa":mean_absolute_error(te[TARGET],yh),
            "RMSE_kPa":mean_squared_error(te[TARGET],yh)**0.5,
            "R2":r2_score(te[TARGET],yh)})
    oof=pd.concat(preds,ignore_index=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.output_dir,
        "4_4_single_input_metrics_by_scenario.csv"),index=False,encoding="utf-8-sig")
    oof.to_csv(os.path.join(a.output_dir,
        "4_4_single_input_LOSO_predictions.csv"),index=False,encoding="utf-8-sig")
    pooled={"n":len(oof),
        "MAE_kPa":mean_absolute_error(oof.y_true_kpa,oof.y_pred_kpa),
        "RMSE_kPa":mean_squared_error(oof.y_true_kpa,oof.y_pred_kpa)**0.5,
        "R2":r2_score(oof.y_true_kpa,oof.y_pred_kpa),
        "negative_prediction_count":int((oof.y_pred_kpa<0).sum())}
    pd.DataFrame([pooled]).to_csv(os.path.join(a.output_dir,
        "4_4_single_input_overall_metrics.csv"),index=False,encoding="utf-8-sig")
    final=XGBRegressor(**PARAMS).fit(df[[FEATURE]],df[TARGET],verbose=False)
    final.save_model(os.path.join(a.output_dir,
        "4_4_single_input_xgboost_full_dataset.json"))
    pd.DataFrame({"feature":[FEATURE],
        "importance":final.feature_importances_}).to_csv(os.path.join(a.output_dir,
        "4_4_single_input_feature_importance.csv"),index=False,encoding="utf-8-sig")
    print(pd.DataFrame([pooled]).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
