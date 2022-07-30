FROM python:3.9-slim

WORKDIR /wrk

EXPOSE 5000

COPY requirements.txt ./
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

# I ignore lots of unnecessary things in .dockerignore
COPY . .

CMD [ "python", "./app.py" ]