import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.models import HoverTool,ColumnDataSource, Select, Range1d, LinearAxis
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models.annotations import Label
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import shap
from shap.plots import _waterfall
shap.initjs()
from dashboard_functions import *
import pickle
import time
import sys
import os

st.set_page_config(layout="wide")

st.title("Get AI advice on your client credit application")

st.header("Upload client's data")

############## DATA PROCESSING ##############

#### LOADING DATA ####

# Get the pickled preprocessor, model, explainer
pickle_cat_to_num = open("cat_to_num.pickle", "rb")
cat_to_num_dict = pickle.load(pickle_cat_to_num)
pickle_preprocessor = open("scalers_preprocessing.pickle", "rb")
scalers_preprocessing = pickle.load(pickle_preprocessor)
pickle_model = open("best_model_lr.pickle", "rb")
best_model_lr = pickle.load(pickle_model)
pickle_explainer = open("kernel_explainer.pickle", "rb")
kernel_explainer = pickle.load(pickle_explainer)
pickle_feature_names = open("feature_names.pickle", "rb")
feature_names = pickle.load(pickle_feature_names)
pickle_all_clients_data = open("all_clients_data.pickle", "rb")
all_clients_data = pickle.load(pickle_all_clients_data)

# Load data from client

st.write(f"Excel client file must contains only one client data with sheets names: 'application', \
        'bureau', 'bureau_balance', 'installments', 'pos_cash', 'previous_app'.")
uploaded_file = st.file_uploader("Load client Excel file here")

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

#### AGGREGATE DATASETS ####

with st.spinner(f"Preparing data..."):

    ## Transform categorical variables to numerical ##
    bureau_all_num, bureau_balance_all_num, previous_app_all_num, pos_cash_all_num = cat_to_num(bureau, 
                                                                                                bureau_balance, 
                                                                                                previous_app, 
                                                                                                pos_cash, 
                                                                                                cat_to_num_dict)

    ## Feature engineering and aggregation ##
    data_client = main_agg(application, bureau_all_num, bureau_balance_all_num, 
                            previous_app_all_num, installments, pos_cash_all_num)
    # drop unecessary columns
    features_to_drop = ['SK_ID_CURR',
                        'AMT_GOODS_PRICE',
                        'APARTMENTS_AVG',
                        'BASEMENTAREA_AVG',
                        'LIVINGAPARTMENTS_AVG',
                        'LIVINGAREA_AVG',
                        'NONLIVINGAPARTMENTS_AVG',
                        'NONLIVINGAREA_AVG',
                        'APARTMENTS_MODE',
                        'BASEMENTAREA_MODE',
                        'YEARS_BEGINEXPLUATATION_MODE',
                        'YEARS_BUILD_MODE',
                        'COMMONAREA_MODE',
                        'ELEVATORS_MODE',
                        'ENTRANCES_MODE',
                        'FLOORSMAX_MODE',
                        'FLOORSMIN_MODE',
                        'LANDAREA_MODE',
                        'LIVINGAPARTMENTS_MODE',
                        'LIVINGAREA_MODE',
                        'NONLIVINGAPARTMENTS_MODE',
                        'NONLIVINGAREA_MODE',
                        'APARTMENTS_MEDI',
                        'BASEMENTAREA_MEDI',
                        'YEARS_BEGINEXPLUATATION_MEDI',
                        'YEARS_BUILD_MEDI',
                        'COMMONAREA_MEDI',
                        'ELEVATORS_MEDI',
                        'ENTRANCES_MEDI',
                        'FLOORSMAX_MEDI',
                        'FLOORSMIN_MEDI',
                        'LANDAREA_MEDI',
                        'LIVINGAPARTMENTS_MEDI',
                        'LIVINGAREA_MEDI',
                        'NONLIVINGAPARTMENTS_MEDI',
                        'NONLIVINGAREA_MEDI',
                        'FONDKAPREMONT_MODE',
                        'HOUSETYPE_MODE',
                        'TOTALAREA_MODE',
                        'WALLSMATERIAL_MODE',
                        'EMERGENCYSTATE_MODE',
                        'COMMONAREA_AVG',
                        'ELEVATORS_AVG',
                        'ENTRANCES_AVG',
                        'FLOORSMAX_AVG',
                        'FLOORSMIN_AVG',
                        'LANDAREA_AVG',
                        'YEARS_BEGINEXPLUATATION_AVG',
                        'YEARS_BUILD_AVG',
                        'NAME_TYPE_SUITE',
                        'OWN_CAR_AGE',
                        'CNT_FAM_MEMBERS']

    data_client.drop(features_to_drop, axis=1, inplace=True)

    # Re-order columns
    data_client = data_client[['CNT_CHILDREN',
                                'AMT_INCOME_TOTAL',
                                'AMT_CREDIT',
                                'AMT_ANNUITY',
                                'REGION_POPULATION_RELATIVE',
                                'DAYS_BIRTH',
                                'DAYS_EMPLOYED',
                                'DAYS_REGISTRATION',
                                'DAYS_ID_PUBLISH',
                                'REGION_RATING_CLIENT',
                                'REGION_RATING_CLIENT_W_CITY',
                                'HOUR_APPR_PROCESS_START',
                                'REG_REGION_NOT_LIVE_REGION',
                                'REG_REGION_NOT_WORK_REGION',
                                'LIVE_REGION_NOT_WORK_REGION',
                                'REG_CITY_NOT_LIVE_CITY',
                                'REG_CITY_NOT_WORK_CITY',
                                'LIVE_CITY_NOT_WORK_CITY',
                                'EXT_SOURCE_1',
                                'EXT_SOURCE_2',
                                'EXT_SOURCE_3',
                                'OBS_30_CNT_SOCIAL_CIRCLE',
                                'DEF_30_CNT_SOCIAL_CIRCLE',
                                'OBS_60_CNT_SOCIAL_CIRCLE',
                                'DEF_60_CNT_SOCIAL_CIRCLE',
                                'DAYS_LAST_PHONE_CHANGE',
                                'AMT_REQ_CREDIT_BUREAU_HOUR',
                                'AMT_REQ_CREDIT_BUREAU_DAY',
                                'AMT_REQ_CREDIT_BUREAU_WEEK',
                                'AMT_REQ_CREDIT_BUREAU_MON',
                                'AMT_REQ_CREDIT_BUREAU_QRT',
                                'AMT_REQ_CREDIT_BUREAU_YEAR',
                                'DAYS_EMPLOYED_PERC',
                                'INCOME_CREDIT_PERC',
                                'INCOME_PER_PERSON',
                                'ANNUITY_INCOME_PERC',
                                'PAYMENT_RATE',
                                'BUREAU_DAYS_CREDIT_MIN',
                                'BUREAU_DAYS_CREDIT_MAX',
                                'BUREAU_DAYS_CREDIT_MEAN',
                                'BUREAU_DAYS_CREDIT_ENDDATE_MAX',
                                'BUREAU_DAYS_CREDIT_ENDDATE_MEAN',
                                'BUREAU_CREDIT_DAY_OVERDUE_MAX',
                                'BUREAU_AMT_CREDIT_SUM_MAX',
                                'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
                                'BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN',
                                'BUREAU_AMT_CREDIT_SUM_LIMIT_SUM',
                                'BUREAU_CNT_CREDIT_PROLONG_SUM',
                                'BUREAU_MONTHS_BALANCE_SIZE_SUM',
                                'BUREAU_CREDIT_ACTIVE_MEAN',
                                'BUREAU_CREDIT_ACTIVE_SUM',
                                'BUREAU_CREDIT_CURRENCY_MEAN',
                                'BUREAU_CREDIT_TYPE_MEAN',
                                'BUREAU_CREDIT_TYPE_MAX',
                                'PREV_AMT_ANNUITY_MIN',
                                'PREV_AMT_ANNUITY_MAX',
                                'PREV_AMT_ANNUITY_MEAN',
                                'PREV_AMT_APPLICATION_MIN',
                                'PREV_AMT_CREDIT_MIN',
                                'PREV_APP_CREDIT_PERC_MIN',
                                'PREV_APP_CREDIT_PERC_MAX',
                                'PREV_APP_CREDIT_PERC_MEAN',
                                'PREV_AMT_DOWN_PAYMENT_MIN',
                                'PREV_AMT_DOWN_PAYMENT_MEAN',
                                'PREV_AMT_GOODS_PRICE_MIN',
                                'PREV_HOUR_APPR_PROCESS_START_MIN',
                                'PREV_HOUR_APPR_PROCESS_START_MAX',
                                'PREV_HOUR_APPR_PROCESS_START_MEAN',
                                'PREV_DAYS_DECISION_MIN',
                                'PREV_DAYS_DECISION_MAX',
                                'PREV_CNT_PAYMENT_MEAN',
                                'PREV_FLAG_LAST_APPL_PER_CONTRACT_MEAN',
                                'PREV_NAME_CASH_LOAN_PURPOSE_MEAN',
                                'PREV_NAME_CONTRACT_STATUS_MAX',
                                'PREV_CODE_REJECT_REASON_MEAN',
                                'PREV_NAME_YIELD_GROUP_MEAN',
                                'PREV_PRODUCT_COMBINATION_MEAN',
                                'POS_MONTHS_BALANCE_MAX',
                                'POS_MONTHS_BALANCE_MEAN',
                                'POS_SK_DPD_MAX',
                                'POS_SK_DPD_DEF_MAX',
                                'POS_NAME_CONTRACT_STATUS_MIN',
                                'POS_NAME_CONTRACT_STATUS_MAX',
                                'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',
                                'INSTAL_DBD_MAX',
                                'INSTAL_DBD_MEAN',
                                'INSTAL_DBD_SUM',
                                'INSTAL_PAYMENT_PERC_MAX',
                                'INSTAL_PAYMENT_DIFF_MEAN',
                                'INSTAL_AMT_INSTALMENT_MAX',
                                'INSTAL_AMT_PAYMENT_MIN',
                                'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
                                'NAME_CONTRACT_TYPE',
                                'CODE_GENDER',
                                'FLAG_OWN_CAR',
                                'FLAG_OWN_REALTY',
                                'NAME_INCOME_TYPE',
                                'NAME_EDUCATION_TYPE',
                                'NAME_FAMILY_STATUS',
                                'NAME_HOUSING_TYPE',
                                'OCCUPATION_TYPE',
                                'WEEKDAY_APPR_PROCESS_START',
                                'ORGANIZATION_TYPE',
                                'FLAG_MOBIL',
                                'FLAG_EMP_PHONE',
                                'FLAG_WORK_PHONE',
                                'FLAG_CONT_MOBILE',
                                'FLAG_PHONE',
                                'FLAG_EMAIL',
                                'FLAG_DOCUMENT_2',
                                'FLAG_DOCUMENT_3',
                                'FLAG_DOCUMENT_4',
                                'FLAG_DOCUMENT_5',
                                'FLAG_DOCUMENT_6',
                                'FLAG_DOCUMENT_7',
                                'FLAG_DOCUMENT_8',
                                'FLAG_DOCUMENT_9',
                                'FLAG_DOCUMENT_10',
                                'FLAG_DOCUMENT_11',
                                'FLAG_DOCUMENT_12',
                                'FLAG_DOCUMENT_13',
                                'FLAG_DOCUMENT_14',
                                'FLAG_DOCUMENT_15',
                                'FLAG_DOCUMENT_16',
                                'FLAG_DOCUMENT_17',
                                'FLAG_DOCUMENT_18',
                                'FLAG_DOCUMENT_19',
                                'FLAG_DOCUMENT_20',
                                'FLAG_DOCUMENT_21']]

    # Add a line for display
    display_data_client = data_client.append(pd.Series(dtype="int64"), ignore_index=True)

    #### MODELING ####

    # Transform client's data
    data_client_transformed = scalers_preprocessing.transform(data_client)
    data_client_transformed = pd.DataFrame(data_client_transformed)

st.success(f"✅ Data are ready.")

if st.checkbox(f"Show prepared data for client #{client_id}."):
    st.dataframe(display_data_client)

with st.spinner(f"AI at work..."):

    # Predict
    prediction = best_model_lr.predict_proba(data_client_transformed)
    prediction_default = prediction[0][1]*100

    # Explain features
    explained_sample = kernel_explainer.shap_values(data_client_transformed.loc[0, :])
    # Format shap values
    shap_explained, most_important_features = format_shap_values(explained_sample, feature_names)

    #### DISTRIBUTION PLOTS ####
    # Default feature
    num_features = list(data_client.select_dtypes(include=["float64", "int64"]).columns)


############## DISPLAY ##############


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
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = prediction_default,
    mode = "gauge+number",
    title = {'text': "Risk of default (%)"},
    gauge = {'axis': {'range': [None, 100],
                    'tick0': 0,
                    'dtick':10},
            'bar': {'color': 'cornflowerblue',
                    'thickness': 0.3,
                    'line': {'color': 'black'}},
            'steps' : [{'range': [0, 40], 'color': "green"},
                        {'range': [41, 60], 'color': "orange"},
                        {'range': [61, 100], 'color': "red"}]}
             ))

    fig_gauge.update_layout(width=400, 
                            height=300,
                            margin= {'l': 30, 'r': 40, 'b': 10, 't':10})

    st.plotly_chart(fig_gauge)

st.header("Additional informations")

left_column, right_column = st.columns([3, 2])

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
    explained_plot = figure(y_range=most_important_features, title="Most important data in the algorithm decision")

    source = ColumnDataSource(data=shap_explained)
    bars = explained_plot.hbar(y="features", left="left", right="right", height=0.5, color="color", 
                                hover_line_color="black", hover_line_width=2, source=source)

    explained_plot.xaxis.axis_label = "Impact on model output"
    explained_plot.yaxis.axis_label = "Client's informations"
    explained_plot.add_tools(HoverTool(tooltips=[("Importance", "@shap_values")], 
                                renderers = [bars]))

    st.bokeh_chart(explained_plot)

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
    p = figure(title=f"Client value for {feature_distrib} \ncompared to other clients", 
                plot_width=400, plot_height=500)
    qr = p.quad(top="hist", bottom=0, line_color="white", left="edges_left", right="edges_right",
        fill_color="navy", hover_fill_color="orange", alpha=0.5, hover_alpha=1, source=hist_source)

    p.line(x=client_line["x"], y=client_line["y"], line_color="orange", line_width=2, line_dash="dashed")

    label_client = Label(text="Value for client", x=data_client_value, y=max_histogram, text_color="orange",
                        x_offset=-10, y_offset=10)
    hover_tools = HoverTool(tooltips=[("Between:", "@edges_left"), ("and:", "@edges_right"), ("Count:", "@hist")], 
                        renderers = [qr])

    p.xaxis.axis_label = feature_distrib
    p.y_range.start = 0
    p.y_range.range_padding = 0.2
    p.yaxis.axis_label = "Number of clients"
    p.grid.grid_line_color="grey"
    p.xgrid.grid_line_color=None
    p.ygrid.grid_line_alpha=0.5

    p.add_tools(hover_tools)
    p.add_layout(label_client)


    st.bokeh_chart(p)
