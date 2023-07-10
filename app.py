import subprocess
import time
import socket

from os import getcwd, listdir, system

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


db_proc = None


# Initialize ExplainerDashboard via gunicorn
def start_gunicorn_server():
    global db_proc
    db_proc = subprocess.Popen(["gunicorn", "-b", "0.0.0.0:8050", "dashboard:app"])


# See if port 8050 is currently in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0


# release port 8050 from use
def release_port():
    if is_port_in_use(8050):
        system("kill -9 $(pgrep gunicorn)")


# kill the gunicorn subprocess
def stop_gunicorn_server():
    global db_proc
    if db_proc is not None:
        db_proc.kill()
        db_proc = None


if "dashboard.yaml" in listdir(getcwd()+"/xdashboard/"):
    if st.button("Show Dashboard"):
        stop_gunicorn_server()
        release_port()
        time.sleep(5)
        start_gunicorn_server()
        time.sleep(10)
        
        url = "http://localhost:8402"
        st.write(f"Explainer Dashboard is ready at {url} !")
        st.components.v1.iframe(src=url, height=800, scrolling=True)
