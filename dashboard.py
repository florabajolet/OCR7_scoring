import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.models import HoverTool,ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.annotations import Label
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import shap
#import mlflow
from dashboard_functions import *
from mlflow_api import *
import pickle
import time
import sys
import os


#mlflow.set_tracking_uri(https://localhost:5000)

st.set_page_config(layout="wide")

st.title("Get AI advice on your client credit application")

st.header("Upload client's data")


#----------------------------------------------------------------------------------#
#                                 LOADING DATA                                     #
#----------------------------------------------------------------------------------#

# Get the pickled preprocessor, model, explainer

pickle_all_clients_data = open("all_clients_data.pickle", "rb")
all_clients_data = pickle.load(pickle_all_clients_data)

# Load data from client

st.write(f"Excel client file must contains only one client data with sheets names: 'application', \
        'bureau', 'bureau_balance', 'installments', 'pos_cash', 'previous_app'.")
uploaded_file = st.file_uploader("Load client Excel file here", type=["xls", "xlsx"])

if uploaded_file:
    # Read database of clients
    application = pd.read_excel(uploaded_file, sheet_name="application")
    bureau = pd.read_excel(uploaded_file, sheet_name="bureau")
    bureau_balance = pd.read_excel(uploaded_file, sheet_name="bureau_balance")
    installments = pd.read_excel(uploaded_file, sheet_name="installments")
    pos_cash = pd.read_excel(uploaded_file, sheet_name="pos_cash")
    previous_app = pd.read_excel(uploaded_file, sheet_name="previous_app")

    # Drop unecessary columns
    application.drop(["TARGET"], axis=1, inplace=True)

    # Get client ID
    client_id = application.loc[0, "SK_ID_CURR"]

    st.success(f"✅ Client's data loaded.")

    # Option show data
    if st.checkbox(f"Show input data for client #{client_id}."):
        st.write("APPLICATION data")
        st.dataframe(application)
        st.write("BUREAU data")
        st.dataframe(bureau)
        st.write("BUREAU BALANCE data")
        st.dataframe(bureau_balance)
        st.write("PREVIOUS APPLICATION data")
        st.dataframe(previous_app)
        st.write("INSTALLEMENTS data")
        st.dataframe(installments)
        st.write("POS_CASH data")
        st.dataframe(pos_cash)

    #----------------------------------------------------------------------------------#
    #                                 PREPROCESSING                                    #
    #----------------------------------------------------------------------------------#

    with st.spinner(f"Preparing data..."):
        data_client, display_data_client, data_client_transformed, data_client_transformed_json = preprocessing_main(application, 
            bureau, bureau_balance, installments, previous_app, pos_cash)

    st.success(f"✅ Data are ready.")

    if st.checkbox(f"Show prepared data for client #{client_id}."):
        st.dataframe(display_data_client)
    
    #----------------------------------------------------------------------------------#
    #                           PREDICT, EXPLAIN FEATURES                              #
    #----------------------------------------------------------------------------------#

    with st.spinner(f"AI at work..."):

        # Predict
        prediction_default = get_prediction(data_client_transformed)

        # Explain features
        shap_explained, most_important_features = explain_features(data_client_transformed)

        #### DISTRIBUTION PLOTS ####
        # Default feature
        num_features = list(data_client.select_dtypes(include=["float64", "int64"]).columns)


    #----------------------------------------------------------------------------------#
    #                               DISPLAY RESULTS                                    #
    #----------------------------------------------------------------------------------#

    st.write("""---""")
    st.header(f"Our recommandation for client #{client_id}")

    left_column_recom, right_column_recom = st.columns(2)

    with left_column_recom:
        # Display clients data and prediction
        st.write(f"Client #{client_id} has **{prediction_default:.1f} % of risk** to make default.")

        if prediction_default < 30:
            st.write(f"We recommand to **accept** client's application to loan.")
        elif (prediction_default >= 30) & (prediction_default <= 50):
            st.write(f"Client's chances to make default are between 30 and 50% . We recommand \
                        to **analyse closely** the data to make your decision.")
        else:
            st.write(f"We recommand to **reject** client's application to loan.")
        
        st.caption(f"Below 30% of default risk, we recommand to accept client application.\
                    Above 50% of default risk, we recommand to reject client application. \
                    Between 30 and 50%, your expertise will be your best advice in your decision making.")

    with right_column_recom:
        fig_gauge = plot_gauge(prediction_default)
        st.plotly_chart(fig_gauge)

    st.header("Additional informations")

    left_column, right_column = st.columns([3, 2])

    # SHAP feature explainer
    with left_column:

        st.write(f"AI algorithms can be seen as quite a black box. To help you make your decision, \
            here are a few supplementary informations.")

        st.caption(f"This plot show you the 15 data that influenced the most \
                the algorithm decision for your client. Data at the top \
                with the largest values are the most important (they are classified \
                in descending order). Positive (red) values \
                indicate an increase in default risk. On the contrary, negative (green) \
                values indicate an increase in chance of reimboursment.")

        # Bokeh plot explained features
        explained_plot = plot_important_features(shap_explained, most_important_features)
        st.bokeh_chart(explained_plot)

    # Distribution plot
    with right_column:
        feature_distrib = st.selectbox(
            "Select the data your are interested in the the dropdown menu.",
            num_features)
        
        # Get value for client
        data_client_value = data_client.loc[data_client.index[0], feature_distrib]
        # Generate distribution data
        hist, edges = np.histogram(all_clients_data[feature_distrib], bins=50)
        hist_source_df = pd.DataFrame({"edges_left": edges[:-1], "edges_right": edges[1:], "hist":hist})
        max_histogram = hist_source_df["hist"].max()
        client_line = pd.DataFrame({"x": [data_client_value, data_client_value],
                                    "y": [0, max_histogram]})
        hist_source = ColumnDataSource(data=hist_source_df)

        st.caption(f"Here you can compare your client value for a specific data compared to a \
                large pool of 80 000+ clients.  The histogram below show \
                the repartition of values for all clients with your client value highlighted in orange.")

        # Make figure
        distrib = plot_feature_distrib(feature_distrib, client_line, hist_source, data_client_value, max_histogram)
        st.bokeh_chart(distrib)


        test = get_prediction_api(data_client_transformed_json)
        st.write(test)