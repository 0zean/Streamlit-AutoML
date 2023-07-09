FROM python:3.10

EXPOSE 8700

WORKDIR /app

RUN mkdir -p xdashboard
RUN mkdir -p pycaret_assets
RUN mkdir -p pycaret_assets/models
RUN mkdir -p pycaret_assets/experiments

COPY . /app/

RUN pip install -r /app/requirements.txt

CMD streamlit run /app/app.py --server.port 8700 --server.enableCORS true