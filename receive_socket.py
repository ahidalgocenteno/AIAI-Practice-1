import socket
import pickle
import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import time

# Define the server class
class Server:
    def __init__(self, host='localhost', port=80, num_clients=1):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.server_sockets = []
    
    def start(self):
        # Start the server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.num_clients)
        
        print(f"Server started, waiting for {self.num_clients} clients to connect...")
        for _ in range(self.num_clients):
            client, addr = server_socket.accept()
            self.server_sockets.append(client)
            print(f"Client {addr} connected.")

        self.collect_results()
        self.close_connections()
    
    def collect_results(self):
        yolo_net = YOLO("yolov8n.pt")

        for server_socket in self.server_sockets:
            while True:
                data = b""
                result = None

                while True:
                    packet = server_socket.recv(4096)

                    if str.encode("foto") in packet:
                        index = packet.find(str.encode("foto"))
                        packet = packet[:index]
                        data+=packet
                        break
                    else: 
                        data+=packet

                try:
                    result = pickle.loads(data)
                except:
                    server_socket.sendall(str(0).encode('utf8'))
                    continue

                if result is not None:
                    print(f"[INFO] Client {server_socket.getpeername()} Data Received!")
                    print(f"[INFO] Evaluating the data...")

                    # Add a short delay to prevent timing issues
                    time.sleep(0.1)  # Delay of 100 milliseconds

                    detections = yolo_net.predict(source=result)
                    person_count=0

                    for detection in detections[0]:
                        class_id = detection.boxes.cls.item()
                        confidence = detection.boxes.conf.item()

                        # Check if the detected object is a person (class_id 0)
                        if class_id == 0 and confidence > 0.5:
                            person_count = person_count + 1

                    print(f"People Counter: {person_count}\n")
                    server_socket.sendall(str(person_count).encode('utf8'))

                    # Delay before sending the 'done' message
                    time.sleep(0.5)  # Delay of 100 milliseconds

                    # Send 'done' message to client
                    server_socket.sendall(b'done')

    def close_connections(self):
        for server_socket in self.server_sockets:
            server_socket.close()
        print("All connections closed.")

# Start the server
if __name__ == "__main__":
    server = Server()
    server.start()
