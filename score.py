import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'rf_model_est_5.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    predict_map = {0:'setosa', 1:'virginica', 2:'versicolor'}
    data = np.array(json.loads(raw_data)['data']).reshape(-1, 1)
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return predict_map[y_hat.tolist()[0]]
