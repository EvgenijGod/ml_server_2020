FROM python:3.8-slim

COPY server/src/requirements.txt /root/server/src/requirements.txt

RUN chown -R root:root /root/server

WORKDIR /root/server/src
RUN pip3 install -r requirements.txt

COPY server/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP main.py

RUN chmod +x main.py
CMD ["python3", "main.py"]
