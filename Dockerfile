FROM python:3.10

EXPOSE 8700

WORKDIR /app

COPY . /app/

RUN pip install -r /app/requirements.txt

CMD mkdir /app/xdashboard && \
    mkdir /app/pycaret_assets && \
    mkdir /app/pycaret_assets/models && \
    mkdir /app/pycaret_assets/experiments && \
    streamlit run /app/app.py --server.port 8700 --server.enableCORS true