import socket
import pickle
import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import time
from concurrent.futures import ThreadPoolExecutor

# Define the server class
class Server:
    def __init__(self, host='localhost', port=80, num_clients=4, max_workers=8):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.connected_clients = 0
        self.disconnected_clients = 0
        self.server_socket = None
        self.max_workers = max_workers

    
    def start(self):
        # Start the server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_clients)
        
        print(f"Server started, waiting for clients on {self.host}:{self.port}...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                while True:
                    if self.connected_clients == self.num_clients:
                        print('\n[INFO] Maximum number of clients reached. No more connections will be accepted')
                        break

                    # Accept new client connection
                    client_socket, addr = self.server_socket.accept()
                    self.connected_clients += 1
                    print(f"\nClient {addr} connected. Total connected clients: {self.connected_clients}/{self.num_clients}")
                    
                    # Submit the client handler to the executor
                    executor.submit(self.handle_client, client_socket, addr)

                # Wait for all clients to disconnect before shutting down
                while self.disconnected_clients < self.num_clients:
                    time.sleep(1)  # Sleep for a short time to avoid busy waiting

            finally:
                self.server_socket.close()
                print("\n[INFO] Server has shut down.")


    def handle_client(self, client_socket, addr):
        yolo_net = YOLO("yolov8n.pt")
        
        try:
            while True:
                data = b""
                result = None

                while True:
                    packet = client_socket.recv(4096)

                    if not packet:
                        print(f"\n[INFO] Connection closed with client {addr}.")
                        return  
                    
                    if str.encode("foto") in packet:
                        index = packet.find(str.encode("foto"))
                        packet = packet[:index]
                        data += packet
                        break
                    else:
                        data += packet
                
                if not data:
                    break

                try:
                    result = pickle.loads(data)
                except:
                    client_socket.sendall(str(0).encode('utf8'))
                    continue

                if result is not None:
                    print(f"[INFO] Data received from client {addr}, processing...")

                    # Add a short delay to prevent timing issues
                    time.sleep(0.1)  # Delay of 100 milliseconds

                    detections = yolo_net.predict(source=result)
                    person_count = 0
                    for detection in detections[0]:
                        class_id = detection.boxes.cls.item()
                        confidence = detection.boxes.conf.item()

                        if class_id == 0 and confidence > 0.5:
                            person_count += 1

                    # Send the number of people detected to the client
                    client_socket.sendall(str(person_count).encode('utf8'))
                    # Add a short delay to avoid timing issues
                    time.sleep(0.1)
                    # Send a 'done' message
                    client_socket.sendall(b'done')
        
        finally:
            client_socket.close()
            #print("Connection closed with client.")
            self.disconnected_clients += 1
            print(f"[INFO] Client {addr} disconnected. Total disconnected clients: {self.disconnected_clients}/{self.num_clients}")


# Start the server
if __name__ == "__main__":
    server = Server(host='localhost', port=80, num_clients=4, max_workers=8)
    server.start()
