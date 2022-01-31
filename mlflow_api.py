from sklearn.linear_model import LogisticRegression
#import mlflow.sklearn
import requests
import pickle
import json

#mlflow.set_tracking_uri(https://localhost:5000)

def get_prediction_api(data_client_transformed_json):

    mlflow_uri = 'http://127.0.0.1:5000/invocations'

    #pickle_model = open("best_model_lr.pickle", "rb")
    #best_model_lr = pickle.load(pickle_model)
    #best_model_lr = mlflow.sklearn.load_model("mlflow_model")

    headers = {"Content-Type":"application/json"}
    response = requests.post(url=mlflow_uri, 
                            data=data_client_transformed_json,
                            headers=headers)

    prediction = json.loads(response.text)
    #prediction = response.json()
    #prediction = best_model_lr.predict_proba(data_client_transformed)
    #prediction_default=  prediction[0][1]*100

    return prediction
