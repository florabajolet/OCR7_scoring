"""
# My first app
Here's our first attempt at using data to create a table:
"""

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

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

left_column, right_column = st.columns(2)

# Left side:
with left_column:
  add_slider = st.sidebar.slider(
      'Select a range of values',
      0.0, 100.0, (25.0, 75.0))

  st.text_input("Path to data", key="path")
  st.session_state.path

  if st.checkbox('Show dataframe'):
      chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

      chart_data

  chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
  st.write(f"You are in {chosen} house!")

# Right side
with right_column:

  # Add a placeholder
  latest_iteration = st.empty()
  bar = st.progress(0)

  for i in range(100):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)
  '...and now we\'re done!'
  
  x = st.slider('x')  # 👈 this is a widget
  st.write(x, 'squared is', x * x)

  df = pd.DataFrame({
      'first column': [1, 2, 3, 4],
      'second column': [10, 20, 30, 40]
      })

  st.dataframe(df.style.highlight_max(axis=0))


  option = st.selectbox(
      'Which number do you like best?',
      df['first column'])

  'You selected: ', option
