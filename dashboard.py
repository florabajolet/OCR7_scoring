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
from imblearn.over_sampling import SMOTE
import shap
from shap.plots import _waterfall
shap.initjs()
import time
import sys
import os


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

all_data = pd.read_csv(path_data, encoding="utf-8")

all_data.drop(["Unnamed: 0", "SK_ID_CURR", "TARGET"], axis=1, inplace=True)

all_data