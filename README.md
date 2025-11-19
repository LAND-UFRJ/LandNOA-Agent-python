Para rodar:

export HOST_IP=$(hostname -I | awk '{print $1}')
sudo HOST_IP=$HOST_IP docker compose build
sudo docker compose up