import argparse
import mlflow
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path
    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("model_path", model_path)
    
    # Load model from model_path
    model = load_model(model_path + "/models") 
    modelreg = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models",
        registered_model_name=model_name
    )
    client = MlflowClient() 
    model_info = client.get_registered_model(model_name)
    model_version = model_info.latest_versions[0].version
    print(model_info)
    print(model_version)    
    print("Model registered!")

if __name__ == "__main__":
    main()