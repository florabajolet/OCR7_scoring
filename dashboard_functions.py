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
