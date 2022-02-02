# OCR7_scoring
Scoring model on the Home Credit dataset from Kaggle

## Goal of the project
This project aims at analysing various data of credit applications from the US society 'Home Credit' 
and build a model that will predict whether a client will reimbourse the loan of make default.
This is a classification binary problem with unbalanced data (much more reimboursed loan than default).
We are also asked to implement the model into a dashboard that will be used by insurers to help them 
make their decision on credit applications. The dashboard is therefore designed for non data-scientists users.
For this reason, we favor a supervised approach since results are easier to explain than deep-learning models.
An API and technical as well as users documentation complete this project.


## What you need
csv data files can be downloaded at: https://www.kaggle.com/c/home-credit-default-risk/data

Librairies:
- Python 3.8.8
- Matplotlib 3.4.2
- Pandas 1.2.4
- Nympy 1.20.1
- Seaborn 0.11.1
- Bokeh 2.4.1
- Yellowbrick 1.3.post1
- Scikit-learn 0.24.1
- Imblearn 0.8.1
- LightGBM 3.3.2
- Plotly 5.3.1
- Shap 0.40
- Streamlit 1.4.0
- MLflow 1.23.1

## Files
This project include:
* An exploratory data analysis notebook (P7_scoring_EDA.ipynb)
* A modeling notebook (P7_scoring_modeling.ipynb)
* A dashboard implementing model and user-friendly interface for interpretation (dashboard.py and dashboard_functions.py)
* A MLflow API and corresponding model directory (mlflow_api.py and mlflow_pyfunc directory). You can also generate the directory from the last part of the modeling notbook.
* Technical and user documentations (TO DO)

The exploratory analysis and feature aggregation are partly taken from the following Kaggle kernels:
- https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
- https://www.kaggle.com/gpreda/home-credit-default-risk-extensive-eda
- https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features