# ml_server_2020
ml_prarcticum_2020

sudo docker build -t ml_server_2020 .

sudo docker run --rm -p 1234:1234 -v "$PWD/server/data:/root/server/data" ml_server_2020
