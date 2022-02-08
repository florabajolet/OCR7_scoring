#!/bin/sh
sh setup.sh
mlflow models serve -m mlflow_model_pyfunc &
streamlit run dashboard.py