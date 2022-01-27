import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import gc
import os
import sys
from sklearn.preprocessing import StandardScaler


## PREPROCESSING ##
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


## MODELING ##


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
