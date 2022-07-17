FROM python:3.9
WORKDIR /root

COPY ml_app/ ml_app/
RUN pip install --no-cache-dir -r ml_app/requirements.txt

ENV FLASK_APP ml_app/app.py

CMD ["flask", "run", "--host=0.0.0.0"]

