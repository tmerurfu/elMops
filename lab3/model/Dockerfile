FROM python:3.9.19-bullseye

WORKDIR /model

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "model.py"]
