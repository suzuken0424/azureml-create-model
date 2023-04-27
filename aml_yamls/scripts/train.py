import argparse, os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

MAX_MAPE_VALUE = 10 # パーセント

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="input data")
    parser.add_argument("--output_dir", type=str, help="output dir", default="./outputs")
    args = parser.parse_args()
    return args

def main(args):
        
    lines = [
        f"Training data path: {args.input_data}",
        f"output dir path: {args.output_dir}"
    ]
    for line in lines:
        print(line)

    # データ読み込み
    y = pd.read_csv(args.input_data, index_col=0, dtype={1: float}).squeeze("columns")
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of airline passengers"

    y_train, y_test = temporal_train_test_split(y)

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        mlflow.autolog(log_models=False, exclusive=True)
        print('run_id = ', run_id)
        # 学習
        model = ARIMA(
            order=(1, 1, 0),
            seasonal_order=(0, 1, 0, 12),
            suppress_warnings=True
        )
        model.fit(y_train)

        # 評価
        fh = np.arange(1, len(y_test) + 1)
        y_pred = model.predict(fh)
        # calc_mape = MeanAbsolutePercentageError()
        # mape = calc_mape(y_test, y_pred) * 100
        # mlflow.log_metric('mape', mape)
        # mape_check = mape < MAX_MAPE_VALUE

        # モデルの登録
        # if mape_check:
        #     os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        #     mlflow.sklearn.save_model(model, os.path.join(args.output_dir, 'models'))
        #     mlflow.log_artifacts(args.output_dir)

        os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
        mlflow.sklearn.save_model(model, os.path.join(args.output_dir, 'models'))
        mlflow.log_artifacts(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)