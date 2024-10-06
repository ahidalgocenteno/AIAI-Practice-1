import numpy as np
import cv2
from mss import mss
from pynput.mouse import Listener
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep, time
import os
from dotenv import load_dotenv
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

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

class Client:
    def __init__(self):
        self.cont = 0
        self.frame_times = []
        self.people_frames = [] # List to save the people count for each frame
        self.azure_client = None
        self.response_times = []
        
    def start(self):
        # Clear the frames folder before extracting frames
        print('[INFO] Clearing the "frames" folder...')
        clear_frames_folder()

        # Perform frame extraction before connecting to the server
        print('[INFO] Starting frame extraction process...')
        self.frame_times = extract_frames()

        # connect to azure endpoint
        try:
            endpoint = os.getenv('AZURE_ENDPOINT')
            key = os.getenv('AZURE_KEY')
            self.azure_client = ImageAnalysisClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            print("Connected to the server.")
        except Exception as e:
            print(f"Error connecting to the server: {e}")
            return 

        try:
            # Get all images files in the 'frames' folder
            frame_folder = 'frames'
            frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')]

            # Send images to the server
            for i, frame in enumerate(frame_files):
                # Read and serialize the frame (image)
                with open(frame, 'rb') as img_file:
                    img_data = img_file.read()

                print(f'[INFO] Sending frame {frame}...')

                # Send the image to Azure AI for analysis
                send_time = time()
                response = self.azure_client.analyze(
                    img_data,
                    visual_features=[VisualFeatures.PEOPLE],
                    language='en'
                )
                
                # Calculate the response time
                response_time = (time() - send_time)
                self.response_times.append(response_time)
                # go through each response people
                people_counter = sum(1 for person in response.people.list if person.confidence > 0.5)
                self.people_frames.append(people_counter)
                print(f'[INFO] People in frame {i+1}: {people_counter}')
                pass

                sleep(1)

            print(f'People count for every frame: {self.people_frames}')
            

        finally:
            # close the connection
            if self.azure_client:
                self.azure_client.close()

            print('Connection closed.')

            # Save results to CSV and plot graph
            self.save_count_to_csv()
            self.plot_graph()
            self.save_response_times_to_csv()

    def save_response_times_to_csv(self):
        # Save response times to CSV
        csv_data = {
            'Timestamp': self.frame_times,
            'Response_Time': self.response_times
        }
        df = pd.DataFrame(csv_data)

        # Save DataFrame to a CSV file
        csv_filename = 'azure_ai_response_times.csv'
        df.to_csv(csv_filename, index=False)
        print(f'[INFO] Response times saved to {csv_filename}')

    def save_count_to_csv(self):
        # Save data to CSV
        csv_data = {
            'Timestamp': self.frame_times,
            'People_Count': self.people_frames
        }
        df = pd.DataFrame(csv_data)

        # Save DataFrame to a CSV file
        csv_filename = 'azure_ai_people_count.csv'
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
        graph_filename = 'azure_ai_people_count_graph.png'
        plt.savefig(graph_filename)
        print(f'[INFO] Saved graph to {graph_filename}')
        
        # Display the plot
        plt.show()

# Start the client
if __name__ == "__main__":
    load_dotenv()
    azure_vision_client = Client()
    azure_vision_client.start()
