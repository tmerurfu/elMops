FROM python:3.9.19-bullseye

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py"]
