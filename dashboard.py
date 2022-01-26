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

############## DATA PROCESSING ##############

#### LOADING DATA ####

# Load data from client

uploaded_file = st.file_uploader("Upload client Excel file...")

# Get the pickled preprocessor, model, explainer
pickle_cat_to_num = open("cat_to_num.pickle", "rb")
cat_to_num = pickle.load(pickle_cat_to_num)
pickle_preprocessor = open("scalers_preprocessing.pickle", "rb")
scalers_preprocessing = pickle.load(pickle_preprocessor)
pickle_model = open("best_model_lr.pickle", "rb")
best_model_lr = pickle.load(pickle_model)
pickle_explainer = open("kernel_explainer.pickle", "rb")
kernel_explainer = pickle.load(pickle_explainer)
pickle_feature_names = open("feature_names.pickle", "rb")
feature_names = pickle.load(pickle_feature_names)


# Read database of clients
application = pd.read_excel(uploaded_file, sheet_name="application")
bureau = pd.read_excel(uploaded_file, sheet_name="bureau")
bureau_balance = pd.read_excel(uploaded_file, sheet_name="bureau_balance")
installments = pd.read_excel(uploaded_file, sheet_name="installments")
pos_cash = pd.read_excel(uploaded_file, sheet_name="pos_cash")
previous_app = pd.read_excel(uploaded_file, sheet_name="previous_app")


# Drop unecessary columns
application.drop(["SK_ID_CURR", "TARGET"], axis=1, inplace=True)

# to drop "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS",

#### TRANSFORM CATEGORICAL VARIABLES TO NUMERICAL ####


# Select one client at random
#rand_client_idx = np.random.randint(0, all_data.shape[0])
#data_client = all_data.loc[rand_client_idx, :]
#data_client_df = pd.DataFrame(data_client).T
# For display (need 2 lines)
#data_clients = all_data.loc[rand_client_idx:rand_client_idx+1, :]
#data_clients_df = pd.DataFrame(data_clients)

#### MODELING ####


# Transform client's data
#data_client_transformed = scalers_preprocessing.transform(data_client_df)
#data_client_transformed = pd.DataFrame(data_client_transformed)

# Predict
#prediction = best_model_lr.predict_proba(data_client_transformed)
#prediction_default = prediction[0][1]*100

# Explain features
#explained_sample = kernel_explainer.shap_values(data_client_transformed.loc[0, :])
# Format shap values
#shap_explained, most_important_features = format_shap_values(explained_sample, feature_names)

#### DISTRIBUTION PLOTS ####
# Default feature
#num_features = list(all_data.select_dtypes(include=["float64", "int64"]).columns)


############## DISPLAY ##############


# Option show data
if st.checkbox('Show input client data'):
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

# Display clients data and prediction
#st.write(f"Client #{rand_client_idx}")
#st.dataframe(data_clients_df)

#st.write(f"Client #{rand_client_idx} has {prediction_default:.1f} % of risk to make default.")

#if prediction_default < 40:
#    st.write(f"We recommand to accept client's application to loan.")
#elif (prediction_default >= 40) & (prediction_default <= 60):
#    st.write(f"Client's chances to make default are close to 50/50. We recommand to analyse closely the data to make your decision.")
#else:
#    st.write(f"We recommand to reject client's application to loan.")

#left_column, right_column = st.columns([3, 2])

#with left_column:

    # Bokeh plot explained features
    #explained_plot = figure(y_range=most_important_features, title="Most important data in the algorithm decision")

    #source = ColumnDataSource(data=shap_explained)
    #bars = explained_plot.hbar(y="features", left="left", right="right", height=0.5, color="color", 
    #                            hover_line_color="black", hover_line_width=2, source=source)

    #explained_plot.xaxis.axis_label = "Impact on model output"
    #explained_plot.yaxis.axis_label = "Client's informations"
    #explained_plot.add_tools(HoverTool(tooltips=[("Importance", "@shap_values")], 
    #                            renderers = [bars]))

    #st.bokeh_chart(explained_plot)

#with right_column:
    #feature_distrib = st.selectbox(
    #    "Variable distribution to show",
    #    num_features)
    
    # Get value for client
    #data_client_value = data_client_df.loc[data_client_df.index[0], feature_distrib]
    # Generate distribution data
    #hist, edges = np.histogram(all_data[feature_distrib], bins=50)
    #hist_source_df = pd.DataFrame({"edges_left": edges[:-1], "edges_right": edges[1:], "hist":hist})
    #max_histogram = hist_source_df["hist"].max()
    #client_line = pd.DataFrame({"x": [data_client_value, data_client_value],
    #                            "y": [0, max_histogram]})
    #hist_source = ColumnDataSource(data=hist_source_df)

    # Make figure
    #p = figure(title=f"Client value for {feature_distrib} compared to other clients", plot_width=400, plot_height=500)
    #qr = p.quad(top="hist", bottom=0, line_color="white", left="edges_left", right="edges_right",
    #    fill_color="navy", hover_fill_color="orange", alpha=0.5, hover_alpha=1, source=hist_source)

    #p.line(x=client_line["x"], y=client_line["y"], line_color="orange", line_width=2, line_dash="dashed")

    #label_client = Label(text="Value for client", x=data_client_value, y=max_histogram, text_color="orange",
    #                    x_offset=-10, y_offset=10)
    #hover_tools = HoverTool(tooltips=[("Between:", "@edges_left"), ("and:", "@edges_right"), ("Count:", "@hist")], 
    #                    renderers = [qr])

    #p.xaxis.axis_label = feature_distrib
    #p.y_range.start = 0
    #p.y_range.range_padding = 0.2
    #p.yaxis.axis_label = "Number of clients"
    #p.grid.grid_line_color="grey"
    #p.xgrid.grid_line_color=None
    #p.ygrid.grid_line_alpha=0.5

    #p.add_tools(hover_tools)
    #p.add_layout(label_client)


    #st.bokeh_chart(p)
