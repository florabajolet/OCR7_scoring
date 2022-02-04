import pandas as pd
import numpy as np
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.annotations import Label
import plotly.graph_objects as go
import pickle
import shap
shap.initjs()
import streamlit as st

#----------------------------------------------------------------------------------#
#                                 PREPROCESSING                                    #
#----------------------------------------------------------------------------------#

def cat_to_num(bureau, bureau_balance, previous_app, pos_cash, cat_to_num_dict):
    """
    Replace categorical features by ordinal or numerical ones using provided dictionnary for mapping.
    Return processed dataframes.

    bureau, bureau_balance, previous_app, pos_cash: dataframes.
    cat_to_num_dict: dict of nested dict with numerical values to replace categorical classes.
    """
    bureau = bureau.copy()
    bureau_balance = bureau_balance.copy()
    previous_app = previous_app.copy()
    pos_cash = pos_cash.copy()

    # Bureau
    features = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]
    scores = [cat_to_num_dict["categ_b_credit_active"],
            cat_to_num_dict["categ_b_currency"], 
            cat_to_num_dict["score_credit_type"]]

    for feature, score in zip(features, scores):
        bureau[feature] = bureau[feature].map(score)
    
    # Bureau balance
    bureau_balance["STATUS"] = bureau_balance["STATUS"].map(cat_to_num_dict["categ_bb_status"])

    #Previous app
    features = ["NAME_CONTRACT_TYPE", 
                "NAME_CONTRACT_STATUS", 
                "NAME_YIELD_GROUP", 
                "NAME_PRODUCT_TYPE", 
                "FLAG_LAST_APPL_PER_CONTRACT", 
                "CODE_REJECT_REASON", 
                "NAME_CASH_LOAN_PURPOSE", 
                "PRODUCT_COMBINATION", 
                "CHANNEL_TYPE", 
                "NAME_SELLER_INDUSTRY", 
                "NAME_GOODS_CATEGORY"]
    scores = [cat_to_num_dict["categ_prev_name_contract_type"], 
            cat_to_num_dict["categ_prev_name_contract_status"],
            cat_to_num_dict["categ_prev_name_yield_group"], 
            cat_to_num_dict["categ_prev_name_product_type"], 
            cat_to_num_dict["categ_prev_flag_last_appl_per_contract"],
            cat_to_num_dict["score_code_reject_reason"],
            cat_to_num_dict["score_name_cash_loan_purpose"],
            cat_to_num_dict["score_product_combination"],
            cat_to_num_dict["score_channel_type"],
            cat_to_num_dict["score_name_seller_industry"],
            cat_to_num_dict["score_name_goods_category"]]

    # Replace classes by ordinal or numerical value
    for feature, score in zip(features, scores):
        previous_app[feature] = previous_app[feature].map(score)

    # Drop categorical columns with little influence on default risk
    previous_app.drop(["WEEKDAY_APPR_PROCESS_START", "NAME_PAYMENT_TYPE", 
                                "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO"], 
                                axis=1, inplace=True)

    # Pos_cash
    pos_cash["NAME_CONTRACT_STATUS"] = pos_cash["NAME_CONTRACT_STATUS"] \
                                                .map(cat_to_num_dict["categ_pos_name_contract_status"])

    return bureau, bureau_balance, previous_app, pos_cash


def application_agg(df, verbose=False):
    """
    Create new features on the application dataframe and replace DAYS_EMPLOYED by NaN if needed.
    Return dataframe with new features.
    Optional: verbose to print info for debug.
    """

    df_new = df.copy()

    # Replace NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df_new['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df_new['DAYS_EMPLOYED_PERC'] = df_new['DAYS_EMPLOYED'] / df_new['DAYS_BIRTH']
    df_new['INCOME_CREDIT_PERC'] = df_new['AMT_INCOME_TOTAL'] / df_new['AMT_CREDIT']
    df_new['INCOME_PER_PERSON'] = df_new['AMT_INCOME_TOTAL'] / df_new['CNT_FAM_MEMBERS']
    df_new['ANNUITY_INCOME_PERC'] = df_new['AMT_ANNUITY'] / df_new['AMT_INCOME_TOTAL']
    df_new['PAYMENT_RATE'] = df_new['AMT_ANNUITY'] / df_new['AMT_CREDIT']
    
    if verbose:
        print(f"FINAL SHAPE: n_rows: {df_new.shape[0]} / n_columns: {df_new.shape[1]}.")

    return df_new


def bureau_and_balance_agg(df_bureau, df_bureau_balance, verbose=False):
    """
    Aggregate bureau and bureau_balance data with feature engineering.
    Return new dataframe.
    """

    df_bureau_new = df_bureau.copy()
    df_bureau_balance_new = df_bureau_balance.copy()
    
    # Bureau balance: Perform aggregations and merge with bureau
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size'],
                      "STATUS": ["mean"]}

    if verbose:
        print("Aggregate bureau_balance...")
    bb_agg = df_bureau_balance_new.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    if verbose:
        print("Join bureau and bureau_balance...")
    df_bureau_new = df_bureau_new.join(bb_agg, how='left', on='SK_ID_BUREAU')
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'AMT_CREDIT_SUM': ['max'],
        'AMT_CREDIT_SUM_DEBT': ['mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_SIZE': ['sum']
    }
    
    # Categorical (ordinal) aggregations
    cat_aggregations = {
        'CREDIT_ACTIVE': ['mean', 'sum'],
        'CREDIT_CURRENCY': ['mean'],
        'CREDIT_TYPE': ['mean', 'max']
    }
    
    if verbose:
        print("Aggregate bureau...")
    bureau_agg = df_bureau_new.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    return bureau_agg


def previous_applications_agg(df, verbose=False):
    """
    Replace false values by NaN.
    Aggregate previous applications dataframe with feature engineering.
    Return new dataframe.
    """
    df_prev_app = df.copy()

    if verbose:
        print("Feature engineering...")
    # Days 365.243 values -> nan
    df_prev_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    df_prev_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    df_prev_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    df_prev_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    df_prev_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    df_prev_app['APP_CREDIT_PERC'] = df_prev_app['AMT_APPLICATION'] / df_prev_app['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min'],
        'AMT_CREDIT': ['min'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min','mean'],
        'AMT_GOODS_PRICE': ['min'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max'],
        'CNT_PAYMENT': ['mean'],
    }
    # Previous applications categorical features
    cat_aggregations = {
        'FLAG_LAST_APPL_PER_CONTRACT': ['mean'],
        'NAME_CASH_LOAN_PURPOSE': ['mean'],
        'NAME_CONTRACT_STATUS': ['max'],
        'CODE_REJECT_REASON': ['mean'],
        'NAME_YIELD_GROUP': ['mean'],
        'PRODUCT_COMBINATION': ['mean'],
    }

    if verbose:
        print("Aggregation of features...")
    prev_agg = df_prev_app.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    return prev_agg

def pos_cash_agg(df):
    """
    Aggregate pos_cash dataframe with feature engineering.
    Return new dataframe.
    """

    df_pos_cash = df.copy()

    #df_pos_cash.drop(["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_ID_PREV"], axis=1, inplace=True)

    # Numerical features aggregation
    num_aggregations = {
        'MONTHS_BALANCE': ['max', 'mean'],
        'SK_DPD': ['max'],
        'SK_DPD_DEF': ['max']
    }

    # categorical feature aggregation
    cat_aggregations = {
        'NAME_CONTRACT_STATUS': ['min', 'max']
    }
    
    pos_agg = df_pos_cash.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    return pos_agg


# Preprocess df_installtallments_payments.csv
def installments_agg(df, verbose=False):
    """
    Aggregate installments dataframe with feature engineering.
    Return new dataframe.
    """
    df_install = df.copy()

    if verbose:
        print("Feature engineering...")
    # Percentage and difference paid in each df_installtallment (amount paid and df_installtallment value)
    df_install['PAYMENT_PERC'] = df_install['AMT_PAYMENT'] / df_install['AMT_INSTALMENT']
    df_install['PAYMENT_DIFF'] = df_install['AMT_INSTALMENT'] - df_install['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    df_install['DBD'] = df_install['DAYS_INSTALMENT'] - df_install['DAYS_ENTRY_PAYMENT']
    df_install['DBD'] = df_install['DBD'].apply(lambda x: x if x > 0 else 0)

    if verbose:
        print("Aggregation...")
    # Features: Perform aggregations
    num_aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max'],
        'PAYMENT_DIFF': ['mean'],
        'AMT_INSTALMENT': ['max'],
        'AMT_PAYMENT': ['min'],
        'DAYS_ENTRY_PAYMENT': ['max']
    }

    df_install_agg = df_install.groupby('SK_ID_CURR').agg(num_aggregations)
    df_install_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in df_install_agg.columns.tolist()])

    return df_install_agg

def main_agg(application, bureau, bureau_balance, previous_app, installments, pos_cash):
    """
    Create new features with internal aggregation of dataframe and join results into a single df.
    Return aggregated dataframe.
    """
    # Feature engineering
    application_fe = application_agg(application)
    bureau_and_balance_fe = bureau_and_balance_agg(bureau, bureau_balance)
    previous_app_fe = previous_applications_agg(previous_app)
    pos_cash_fe = pos_cash_agg(pos_cash)
    installments_fe = installments_agg(installments)

    # Joining all datasets
    application_fe = application_fe.join(bureau_and_balance_fe, how='left', on='SK_ID_CURR')
    application_fe = application_fe.join(previous_app_fe, how='left', on='SK_ID_CURR')
    application_fe = application_fe.join(pos_cash_fe, how='left', on='SK_ID_CURR')
    application_fe = application_fe.join(installments_fe, how='left', on='SK_ID_CURR')

    return application_fe

def preprocessing_main(application, bureau, bureau_balance, installments, previous_app, pos_cash):
    """ 
    All preprocessing of data, including aggregation, scaling of numerical variables and
    transformation of categorical variable.
    Return 3 dataframes: the aggregated data, the aggregated data with a second line for display,
    the transformed client's data to feed the model.
    """
    # Get the pickled dictionnary and preprocessing
    pickle_cat_to_num = open("pkl/cat_to_num.pickle", "rb")
    cat_to_num_dict = pickle.load(pickle_cat_to_num)
    pickle_preprocessor = open("pkl/scalers_preprocessing.pickle", "rb")
    scalers_preprocessing = pickle.load(pickle_preprocessor)

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

    ## Transform features ##
    data_client_transformed = scalers_preprocessing.transform(data_client)
    data_client_transformed = pd.DataFrame(data_client_transformed)
    data_client_transformed_json = data_client_transformed.to_json(orient="split")

    return data_client, display_data_client, data_client_transformed, data_client_transformed_json

def transform_data(data_client):
    """ 
    Scale numerical features and transform categorical features for data 
    coming from clients database.
    """

    pickle_preprocessor = open("pkl/scalers_preprocessing.pickle", "rb")
    scalers_preprocessing = pickle.load(pickle_preprocessor)

    #data_client = pd.DataFrame(data_client).T
    data_client_transformed = scalers_preprocessing.transform(data_client)
    data_client_transformed = pd.DataFrame(data_client_transformed)
    data_client_transformed_json = data_client_transformed.to_json(orient="split")

    return data_client_transformed, data_client_transformed_json

#----------------------------------------------------------------------------------#
#                                 MODELING                                         #
#----------------------------------------------------------------------------------#

def format_shap_values(shap_values, feature_names):
    """
    Format shap values into a dataframe to be plotted with Bokeh.
    Return dataframe with first 15 most important shap values, left and right values and color for Bokeh plot.
    """
    # Formatting df
    df = pd.DataFrame(shap_values, index=feature_names).reset_index()
    df.rename(columns={"index": "features", 0:"shap_values"}, inplace=True)
    df["absolute_values"]=abs(df["shap_values"])
    df.sort_values(by="absolute_values", ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    # Getting left and right from shap
    df["left"] = df["shap_values"].where(df["shap_values"]<0, 0)
    df["right"] = df["shap_values"].where(df["shap_values"]>0, 0)

    # Color depending on sign
    df["color"] = np.where(df["shap_values"]>0, "#D73027", "#1A9851")

    # Select first 15
    shap_explained = df.loc[0:14, ["features", "shap_values", "left", "right", "color"]]
    shap_explained.reset_index(inplace=True)

    # Make list of most important features (inversed for Bokeh)
    most_important_features = shap_explained["features"].tolist()
    most_important_features = most_important_features[::-1]

    return shap_explained, most_important_features


def explain_features(data_client_transformed):
    """
    Load pickeled explainer and names of transformed feature, compute the shap values.
    """
    pickle_explainer = open("pkl/kernel_explainer.pickle", "rb")
    kernel_explainer = pickle.load(pickle_explainer)
    pickle_feature_names = open("pkl/feature_names.pickle", "rb")
    feature_names = pickle.load(pickle_feature_names)

    explained_sample = kernel_explainer.shap_values(data_client_transformed.loc[0, :])
    # Format shap values
    shap_explained, most_important_features = format_shap_values(explained_sample, feature_names)

    return shap_explained, most_important_features


#----------------------------------------------------------------------------------#
#                                     FIGURES                                      #
#----------------------------------------------------------------------------------#


def plot_gauge(prediction_default):
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
            'steps' : [{'range': [0, 30], 'color': "green"},
                        {'range': [31, 50], 'color': "orange"},
                        {'range': [51, 100], 'color': "red"}]}
            ))

    fig_gauge.update_layout(width=400, 
                            height=300,
                            margin= {'l': 30, 'r': 40, 'b': 10, 't':10})
    
    return fig_gauge


def plot_important_features(shap_explained, most_important_features):

    explained_plot = figure(y_range=most_important_features, title="Most important data in the algorithm decision")

    source = ColumnDataSource(data=shap_explained)
    bars = explained_plot.hbar(y="features", left="left", right="right", height=0.5, color="color", 
                                hover_line_color="black", hover_line_width=2, source=source)

    explained_plot.xaxis.axis_label = "Impact on model output"
    explained_plot.yaxis.axis_label = "Client's informations"
    explained_plot.add_tools(HoverTool(tooltips=[("Importance", "@shap_values")], 
                                renderers = [bars]))
    return explained_plot

def plot_feature_distrib(feature_distrib, client_line, hist_source, data_client_value, max_histogram):
    distrib = figure(title=f"Client value for {feature_distrib} \ncompared to other clients", 
                    plot_width=400, plot_height=500)
    qr = distrib.quad(top="hist", bottom=0, line_color="white", left="edges_left", right="edges_right",
        fill_color="navy", hover_fill_color="orange", alpha=0.5, hover_alpha=1, source=hist_source)

    distrib.line(x=client_line["x"], y=client_line["y"], line_color="orange", line_width=2, line_dash="dashed")

    label_client = Label(text="Value for client", x=data_client_value, y=max_histogram, text_color="orange",
                        x_offset=-10, y_offset=10)
    hover_tools = HoverTool(tooltips=[("Between:", "@edges_left"), ("and:", "@edges_right"), ("Count:", "@hist")], 
                        renderers = [qr])

    distrib.xaxis.axis_label = feature_distrib
    distrib.y_range.start = 0
    distrib.y_range.range_padding = 0.2
    distrib.yaxis.axis_label = "Number of clients"
    distrib.grid.grid_line_color="grey"
    distrib.xgrid.grid_line_color=None
    distrib.ygrid.grid_line_alpha=0.5

    distrib.add_tools(hover_tools)
    distrib.add_layout(label_client)

    return distrib