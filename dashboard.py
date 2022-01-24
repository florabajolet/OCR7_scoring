import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import shap
from shap.plots import _waterfall
shap.initjs()
import pickle
import time
import sys
import os

############## DATA PROCESSING ##############

#### LOADING DATA ####

# Load data from client
st.text_input("Path to data", key="path")
st.write("Path: ", st.session_state.path)


#path_application = os.path.join(st.session_state.path, "application_train.csv")
#path_bureau = os.path.join(st.session_state.path, "bureau.csv")
#path_bureau_bal = os.path.join(st.session_state.path, "bureau_balance.csv")
#path_credit_card = os.path.join(st.session_state.path, "credit_card_balance.csv")
#path_installements= os.path.join(st.session_state.path, "installments_payments.csv")
#path_pos_cash = os.path.join(st.session_state.path, "POS_CASH_balance.csv")
#path_previous_app = os.path.join(st.session_state.path, "previous_application.csv")

path_data = os.path.join(st.session_state.path, "data_train_clean_select.csv")

# Add a placeholder
#latest_iteration = st.empty()
#bar = st.progress(0)

# Read database of clients
all_data = pd.read_csv(path_data, encoding="utf-8")

# Drop unecessary columns
all_data.drop(["Unnamed: 0", "SK_ID_CURR", "TARGET"], axis=1, inplace=True)

# Select one client at random
rand_client_idx = np.random.randint(0, all_data.shape[0])
data_client = all_data.loc[rand_client_idx, :]
data_client_df = pd.DataFrame(data_client).T
# For display (need 2 lines)
data_clients = all_data.loc[rand_client_idx:rand_client_idx+1, :]
data_clients_df = pd.DataFrame(data_clients)

#### MODELING ####

# Get the pickled preprocessor, model, explainer
pickle_preprocessor = open("scalers_preprocessing.pickle", "rb")
scalers_preprocessing = pickle.load(pickle_preprocessor)
pickle_model = open("best_model_lr.pickle", "rb")
best_model_lr = pickle.load(pickle_model)
pickle_explainer = open("kernel_explainer.pickle", "rb")
kernel_explainer = pickle.load(pickle_explainer)
pickle_feature_names = open("feature_names.pickle", "rb")
feature_names = pickle.load(pickle_feature_names)

# Transform client's data
data_client_transformed = scalers_preprocessing.transform(data_client_df)
data_client_transformed = pd.DataFrame(data_client_transformed)

# Predict
prediction = best_model_lr.predict_proba(data_client_transformed)
prediction_default = prediction[0][1]*100

# Explain features
explained_sample = kernel_explainer.shap_values(data_client_transformed.loc[0, :])
# Format shap values
explained_sample_df = pd.DataFrame(explained_sample, index=feature_names).reset_index()
explained_sample_df.rename(columns={"index": "features", 0:"shap_values"}, inplace=True)
explained_sample_df["absolute_values"]=abs(explained_sample_df["shap_values"])
explained_sample_df.sort_values(by="absolute_values", ascending=False, inplace=True)
explained_sample_df.reset_index(inplace=True)
explained_sample_df.drop("index", axis=1, inplace=True)

############## DISPLAY ##############

# Display all data if necessary
if st.checkbox('Show full dataframe'):
    all_data

# Display clients data and prediction
st.write(f"Client #{rand_client_idx}")
st.dataframe(data_clients_df)

st.write(f"Client #{rand_client_idx} has {prediction_default:.1f} % of risk to make default.")

if prediction_default < 40:
    st.write(f"We recommand to accept client's application to loan.")
elif (prediction_default >= 40) & (prediction_default <= 60):
    st.write(f"Client's chances to make default are close to 50/50. We recommand to analyse losely the data to make your decision.")
else:
    st.write(f"We recommand to reject client's application to loan.")

# Explained features
shap.bar_plot(explained_sample,
              feature_names=feature_names,
              max_display=15)
