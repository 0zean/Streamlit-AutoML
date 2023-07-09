import subprocess
import time
from os import getcwd, listdir

import streamlit as st

from AutoML import AutoML
from DataFrameLoad import DataFrameLoad
from ExplainerBuilder import ExplainerBuilder

st.title("Classification AutoML")

data_source = st.radio("Select data source:", ("CSV File", "SQL Database"))

# DataFrameLoad class
if data_source is not None:
    df_load = DataFrameLoad(data_source)
    df_load.create_form()

# AutoML 
exp_name = st.text_input("Type in an experiment name to track: ")

target_name = st.text_input("Type in an output variable: ")

title = st.text_input("Type in a title for your dashboard: ")

if st.button("Run Experiment"):
    automl = AutoML(df=df_load.retrieve_df(),
                    target=target_name,
                    experiment_name=exp_name)
    
    automl.automate()

    model_path = f"{getcwd()}/pycaret_assets/models/{automl.model_name}"
    test_path = automl.test_data
    x = automl.xdata
    y = automl.ydata

    st.write("Building Explainer Dashboard... ")
    ExplainerBuilder(model_path=model_path,
                     x_test=x,
                     y_test=y,
                     target=target_name,
                     title=title).dashboard_save()

# Set up for Linux
if "dashboard.yaml" in listdir(getcwd()+"/xdashboard/"):
    if st.button("Show Dashboard"):
        db_proc = subprocess.Popen(["gunicorn", "-w", "3", "-b", "localhost:8050", "dashboard:app"],
                                stdout=subprocess.PIPE, cwd=getcwd())
        time.sleep(12)
        
        url = "http://localhost:8050"
        st.write(f"Explainer Dashboard is ready at {url} !")
        st.components.v1.iframe(src=url, height=800, scrolling=True)
