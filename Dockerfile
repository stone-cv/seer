FROM python:3.9
ENV CONFIG src/config.yaml

WORKDIR /code/

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install pipenv
COPY Pipfile Pipfile.lock /code/
RUN pipenv install --system --dev

COPY . /code/

ENTRYPOINT ["python", "src/main.py"]