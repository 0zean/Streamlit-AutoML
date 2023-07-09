FROM python:3.10

EXPOSE 8700

WORKDIR /app

COPY . /app/

RUN mkdir xdashboard

RUN mkdir -p pycaret_assets/experiments pycaret_assets/models

RUN pip install -r /app/requirements.txt

CMD streamlit run /app/app.py --server.port 8700 --server.enableCORS true