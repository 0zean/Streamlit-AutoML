FROM python:3.10

EXPOSE 8700 8050

WORKDIR /app

COPY . /app/

RUN pip install -r /app/requirements.txt

CMD mkdir -p /app/xdashboard && \
    mkdir -p /app/pycaret_assets && \
    mkdir -p /app/pycaret_assets/models && \
    mkdir -p /app/pycaret_assets/experiments && \
    streamlit run /app/app.py --server.port 8700 --server.enableCORS true