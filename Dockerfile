FROM python:3.8
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run --server.port $PORT app.py