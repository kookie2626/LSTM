import os
import pickle
import time
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# 기존 코드 재사용 (데이터 로드 부분)
from project.ML.scripts.train.train_all_meters import load_weather, load_meter, build_features, make_splits, make_sequences, SEQ_LEN, LSTMForecaster

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_BASE = Path("outputs/models/meters")

def update_meter_wape(meter_urn, weather, run_id, client):
    out_dir = OUT_BASE / meter_urn.replace(".", "_")
    if not (out_dir / "vmd_lstm.pt").exists():
        return None

    # Load model & config
    checkpoint = torch.load(out_dir / "vmd_lstm.pt", map_location=DEVICE, weights_only=False)
    conf = checkpoint["model_config"]
    model = LSTMForecaster(conf["input_dim"], conf["hidden_dim"], conf["num_layers"], conf["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    # Load scaler
    with open(out_dir / "vmd_lstm_scaler.pkl", "rb") as f:
        scalers = pickle.load(f)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    # Load data
    meter_s = load_meter(meter_urn)
    df = weather.copy()
    df["target_P"] = meter_s.reindex(df.index).fillna(0)
    df, feat_cols = build_features(df)
    _, val, test = make_splits(df)

    X_val = scaler_X.transform(val[feat_cols])
    y_val = scaler_y.transform(val[["target_P"]]).ravel()
    
    if len(test) > SEQ_LEN:
        X_test = scaler_X.transform(test[feat_cols])
        y_test = scaler_y.transform(test[["target_P"]]).ravel()
    else:
        X_test, y_test = np.zeros((0, len(feat_cols))), np.zeros(0)

    def predict(X_seq):
        seqs = np.array([X_seq[i: i + SEQ_LEN] for i in range(len(X_seq) - SEQ_LEN)], dtype=np.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), 512):
                preds.append(model(torch.from_numpy(seqs[i:i+512]).to(DEVICE)).cpu().numpy())
        return np.concatenate(preds)

    def inv(arr, sc): return sc.inverse_transform(arr.reshape(-1, 1)).ravel()

    # Calculate WAPE for Val
    yhat_val = np.maximum(inv(predict(X_val), scaler_y), 0)
    ytru_val = inv(y_val[SEQ_LEN:], scaler_y)
    
    val_sum = float(np.sum(ytru_val))
    val_wape = float(np.sum(np.abs(ytru_val - yhat_val)) / val_sum * 100) if val_sum > 0 else float("nan")

    # Calculate WAPE for Test
    test_wape = float("nan")
    if len(X_test) > SEQ_LEN:
        yhat_te = np.maximum(inv(predict(X_test), scaler_y), 0)
        ytru_te = inv(y_test[SEQ_LEN:], scaler_y)
        test_sum = float(np.sum(ytru_te))
        test_wape = float(np.sum(np.abs(ytru_te - yhat_te)) / test_sum * 100) if test_sum > 0 else float("nan")

    # Update MLflow
    if run_id:
        if not np.isnan(val_wape):
            client.log_metric(run_id, "val_wape", val_wape)
        if not np.isnan(test_wape):
            client.log_metric(run_id, "test_wape", test_wape)

    return {"val_wape": val_wape, "test_wape": test_wape}

def main():
    print("▶ MLflow에서 기존 Run 정보 가져오는 중...")
    client = MlflowClient(tracking_uri="http://121.134.46.24:5000")
    exp = client.get_experiment_by_name("All-Meters")
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
    
    run_dict = {}
    for r in runs:
        if r.info.status == "FINISHED":
            meter_name = r.data.tags.get("mlflow.runName")
            if meter_name:
                run_dict[meter_name] = r.info.run_id

    print("▶ 날씨 데이터 로드 중...")
    weather = load_weather()

    csv_path = "outputs/all_meters_results.csv"
    df = pd.read_csv(csv_path)
    
    if 'val_wape' not in df.columns:
        df['val_wape'] = np.nan
    if 'test_wape' not in df.columns:
        df['test_wape'] = np.nan

    print(f"▶ 총 {len(df)}개 계량기 평가 시작 (디바이스: {DEVICE})...")
    
    start_t = time.time()
    for idx, row in df.iterrows():
        if row['status'] != 'OK':
            continue
        
        meter_urn = row['meter_urn']
        run_id = run_dict.get(meter_urn)
        
        res = update_meter_wape(meter_urn, weather, run_id, client)
        if res:
            df.at[idx, 'val_wape'] = res['val_wape']
            df.at[idx, 'test_wape'] = res['test_wape']
            print(f"[{idx+1:2d}/{len(df)}] {meter_urn:<22} 완료 | Test WAPE: {res['test_wape']:.2f}%")
        else:
            print(f"[{idx+1:2d}/{len(df)}] {meter_urn:<22} 실패 (모델 없음)")

    # CSV 저장 (기존 mape 컬럼 삭제)
    if 'val_mape' in df.columns:
        df = df.drop(columns=['val_mape'])
    if 'test_mape' in df.columns:
        df = df.drop(columns=['test_mape'])
        
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - start_t
    print(f"\n▶ 모든 평가 완료! (소요 시간: {elapsed:.1f}초)")
    print("▶ MLflow 및 outputs/all_meters_results.csv 업데이트 완료.")

if __name__ == "__main__":
    main()
