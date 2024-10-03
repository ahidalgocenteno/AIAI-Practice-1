import socket
import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.mouse import Listener
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep
import os


# Function to clear the 'frames' folder
def clear_frames_folder():
    frame_folder = 'frames'

    if os.path.exists(frame_folder):
        for file in os.listdir(frame_folder):
            file_path = os.path.join(frame_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    #print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("[INFO] Folder cleared.")
    else:
        os.makedirs(frame_folder)  # Create 'frames' folder if it doesn't exist
        print(f"[INFO] Created folder: {frame_folder}")


cont = 0
pos = {"x":[],"y":[]}

def is_clicked(x, y, button, pressed):
    global cont, pos
    if pressed:
        print('Clicked ! ') #in your case, you can move it to some other pos
        pos["x"].append(x)
        pos["y"].append(y)
        cont+=1
        if cont == 2:
            return False # to stop the thread after click
        
def extract_frames():
    frame_times = []

    # Use MSS to capture the screen
    with Listener(on_click=is_clicked) as listener:
        listener.join()

    bounding_box = {'top': pos["y"][0], 'left': pos["x"][0], 'width': pos["x"][1]-pos["x"][0], 'height': pos["y"][1]-pos["y"][0]}
    sct = mss()
    
    last_time = datetime.now()
    frame_count = 0
    
    while True:
        sct_img = sct.grab(bounding_box)
        cv2.imshow('Screen Capture', np.array(sct_img))

        current_time = datetime.now()
        if (current_time - last_time).total_seconds() >= 5:  # Capture a frame every 5 seconds
            frame_name = f'frames/{current_time.strftime("%H-%M-%S")}.png'
            frame_times.append(current_time.strftime("%H:%M:%S"))
            cv2.imwrite(frame_name, np.array(sct_img))
            print(f'Frame captured: {frame_name}')
            frame_count += 1
            last_time = current_time

        # Press 'q' to quit frame capturing
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            print(f"{frame_count} frames captured.")
            break
    
    return frame_times

# Define the client class
class Client:
    def __init__(self, host='20.33.92.38', port=80):
        self.host = host
        self.port = port
        self.close_socket = False
        self.cont = 0
        self.frame_times = []
        self.people_frames = [] # List to save the people count for each frame
        
    def start(self):
        # Clear the frames folder before extracting frames
        print('[INFO] Clearing the "frames" folder...')
        clear_frames_folder()

        # Perform frame extraction before connecting to the server
        print('[INFO] Starting frame extraction process...')
        self.frame_times = extract_frames()

        # Start the client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print("Connected to the server.")

        try:
            # Get all images files in the 'frames' folder
            frame_folder = 'frames'
            frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]

            # Send images to the server
            for i, frame in enumerate(frame_files):
                # Read and serialize the frame (image)
                img = cv2.imread(frame)
                result = pickle.dumps(img)
                print(f'[INFO] Sending frame {frame}...')

                # Send the serialized image to the server
                client_socket.sendall(result)
                client_socket.sendall(str.encode('foto'))         
            
                # Wait for the server's response (number of people)
                data = client_socket.recv(5)
                if not data:
                    print(f"[ERROR] Failed to receive data from server for frame {i}")
                    break

                people_counter = int(data.decode('utf8'))
                self.people_frames.append(people_counter)
                print(f'[INFO] People in frame {i+1}: {people_counter}')

                # Wait for the 'done' message before sending the next frame
                done_message = self.recv_all(client_socket, 4)  # Ensure full "done" message is received
                if done_message.decode('utf8') == 'done':
                    print('[INFO] Server processed the frame. Sending next one...\n')
                else:
                    print(f"[ERROR] Did not receive 'done' message from server for frame {i}")
                    break

                sleep(1)

            #print(f'People count for every frame: {self.people_frames}')
            

        finally:
            client_socket.close()
            print('Connection closed.')

            # Save results to CSV and plot graph
            self.save_to_csv()
            self.plot_graph()
    

    def recv_all(self, socket, length):
        # Ensure receiving the full length of the message.
        data = b''
        while len(data) < length:
            packet = socket.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data


    def save_to_csv(self):
        # Save data to CSV
        csv_data = {
            'Timestamp': self.frame_times,
            'People_Count': self.people_frames
        }
        df = pd.DataFrame(csv_data)

        # Save DataFrame to a CSV file
        csv_filename = 'people_count.csv'
        df.to_csv(csv_filename, index=False)
        print(f'[INFO] Data saved to {csv_filename}')
    

    def plot_graph(self):
        # Number of people vs. time (timestamps)
        plt.figure(figsize=(10, 6))
        plt.plot(self.frame_times, self.people_frames, marker='o', linestyle='-', color='b')
        
        # Formatting the graph
        plt.title('Number of People Detected Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Number of People')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        
        # Limit the number of ticks
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=20))  # Adjust the number of ticks as needed

        plt.tight_layout()  # Adjust layout to ensure everything fits without overlap

        # Save the plot as an image file
        graph_filename = 'people_count_graph.png'
        plt.savefig(graph_filename)
        print(f'[INFO] Saved graph to {graph_filename}')
        
        # Display the plot
        plt.show()

# Start the client
if __name__ == "__main__":
    client = Client()
    client.start()
