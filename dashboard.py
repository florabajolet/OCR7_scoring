import pandas as pd
import streamlit as st
import numpy as np
import pickle
from bokeh.models import HoverTool,ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.annotations import Label
import plotly.graph_objects as go
import shap
from dashboard_functions import *
from mlflow_api import *


st.set_page_config(layout="wide")

st.title("Get AI advice on your client credit application")

load_type = st.radio(label="Please, choose:", options=["Load client data", "Use client database"])
#----------------------------------------------------------------------------------#
#                                 LOADING DATA                                     #
#----------------------------------------------------------------------------------#

# Get the pickled clients data

#pickle_all_clients_data_01 = open("pkl/all_clients_data_01.pickle", "rb")
#all_clients_data_01 = pickle.load(pickle_all_clients_data_01)
#pickle_all_clients_data_02 = open("pkl/all_clients_data_02.pickle", "rb")
#all_clients_data_02 = pickle.load(pickle_all_clients_data_02)

pickle_all_clients_data = open("pkl/all_clients_data.pickle", "rb")
all_clients_data = pickle.load(pickle_all_clients_data)

# Concatenate data, get IDs and features
#all_clients_data = pd.concat([all_clients_data_01, all_clients_data_02])
all_clients_id = list(all_clients_data["SK_ID_CURR"])
num_features = list(all_clients_data.select_dtypes(include=["float64", "int64"]).columns)

#Define a variable to display sidebar only when data are loaded
sidebar = "Off"
#------------------ Load data from client
if load_type=="Load client data":

    st.header("Upload client's data")

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

        # Get client ID and other variables
        client_id = application.loc[0, "SK_ID_CURR"]
        gender = application.loc[0, "CODE_GENDER"]
        family_status = application.loc[0, "NAME_FAMILY_STATUS"]
        loan_type = application.loc[0, "NAME_CONTRACT_TYPE"]
        income_type = application.loc[0, "NAME_INCOME_TYPE"]
        education = application.loc[0, "NAME_EDUCATION_TYPE"]
        occupation_type = application.loc[0, "OCCUPATION_TYPE"]
        credit = str(round(application.loc[0, "AMT_CREDIT"])) + " $"
        annuity = str(round(application.loc[0, "AMT_ANNUITY"])) + " $"
        days_birth = application.loc[0, "DAYS_BIRTH"]
        days_employed = application.loc[0, "DAYS_EMPLOYED"]
        fam_members = int(application.loc[0, "CNT_FAM_MEMBERS"])

        work = income_type + ", " + occupation_type
        age = -int(round(days_birth/365))
        years_work = -int(round(days_employed/365))

        # Display sidebar on
        sidebar = "On"

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
    
        income_per_person = str(round(data_client.loc[0, "INCOME_PER_PERSON"])) + " $"

#------------------ Use client database    
if load_type=="Use client database":

    client_id = st.selectbox(
            "Select your client ID in the dropdown menu.",
            all_clients_id)
    st.write(f"You selected client # {client_id}.")

    data_client_with_id = all_clients_data[all_clients_data["SK_ID_CURR"]==client_id]
    client_index = data_client_with_id.index[0]

    gender = data_client_with_id.loc[client_index, "CODE_GENDER"]
    family_status = data_client_with_id.loc[client_index, "NAME_FAMILY_STATUS"]
    loan_type = data_client_with_id.loc[client_index, "NAME_CONTRACT_TYPE"]
    income_type = data_client_with_id.loc[client_index, "NAME_INCOME_TYPE"]
    education = data_client_with_id.loc[client_index, "NAME_EDUCATION_TYPE"]
    occupation_type = data_client_with_id.loc[client_index, "OCCUPATION_TYPE"]
    credit = str(round(data_client_with_id.loc[client_index, "AMT_CREDIT"])) + " $"
    annuity = str(round(data_client_with_id.loc[client_index, "AMT_ANNUITY"])) + " $"
    days_birth = data_client_with_id.loc[client_index, "DAYS_BIRTH"]
    days_employed = data_client_with_id.loc[client_index, "DAYS_EMPLOYED"]
    fam_members = int(data_client_with_id.loc[client_index, "CNT_CHILDREN"])
    income_per_person = str(round(data_client_with_id.loc[client_index, "INCOME_PER_PERSON"])) + " $"

    work = income_type + ", " + occupation_type
    age = -int(round(days_birth/365))
    years_work = -int(round(days_employed/365))

    # Allow sidebar
    sidebar = "On"

    if st.checkbox(f"Show prepared data for client #{client_id}."):
        st.dataframe(data_client_with_id)
    
    data_client = data_client_with_id.drop("SK_ID_CURR", axis=1)

    data_client_transformed, data_client_transformed_json = transform_data(data_client)

    #----------------------------------------------------------------------------------#
    #                                   SIDEBAR                                        #
    #----------------------------------------------------------------------------------#

# If data are loaded
if sidebar=="On":
    st.sidebar.header(f"Your client")

    st.sidebar.write("**Gender:** ", gender)
    st.sidebar.write("**Family status:** ", family_status)
    st.sidebar.write("**Loan type:** ", loan_type)
    st.sidebar.write("**Professional situation:** ", work)
    st.sidebar.write("**Education:** ", education)

    st.sidebar.write("""---""")
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Household members", value=fam_members)
    col2.metric(label="Age", value=age)
    col3, col4 = st.sidebar.columns(2)
    col3.metric(label="Years worked", value=years_work)
    col4.metric(label="Income per person", value=income_per_person)
    col5, col6 = st.sidebar.columns(2)
    col5.metric(label="Credit", value=credit)
    col6.metric(label="Annuity", value=annuity)

    #----------------------------------------------------------------------------------#
    #                           PREDICT, EXPLAIN FEATURES                              #
    #----------------------------------------------------------------------------------#


if st.checkbox("Get advice") and sidebar=="On":

    with st.spinner(f"AI at work..."):

        prediction_default = get_prediction_api(data_client_transformed_json)

        # Explain features
        shap_explained, most_important_features = explain_features(data_client_transformed)

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

#----------------------------------------------------------------------------------#
#                                    BOTTOM                                        #
#----------------------------------------------------------------------------------#
st.write("---")

col_about, col_FAQ, col_doc, col_contact = st.columns(4)

with col_about:
    st.write("About us")

with col_FAQ:
    st.write("FAQ")

with col_doc:
    st.write("Technical documentation")

with col_contact:
    st.write("Contact")

