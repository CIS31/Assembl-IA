import os
import sys
import cv2
from ultralytics import YOLO
from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import csv  # Import CSV module for saving timeline data
import psycopg2  # Import psycopg2 for PostgreSQL operations

class HumanEmotionAnalyzer:
    def __init__(self, input_folder_images, input_folder_video, output_folder, yolo_model_path, emotion_model_dir, azure_run=False):
        # Configuration
        self.input_folder_images = input_folder_images
        self.input_folder_video = input_folder_video
        self.output_folder = output_folder
        self.azure_run = azure_run

        # Constants for low and high resolution detection
        self.IMG_SIZE_LOW = 1280
        self.IMG_SIZE_HIGH = 1280 * 2
        self.DETECTION_THRESHOLD = 10

        # Load YOLOv8 face detection model
        self.detection_model = YOLO(yolo_model_path)

        # Load emotion recognition model
        self.emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_dir)
        self.pipe = pipeline('image-classification', model=self.emotion_model, feature_extractor=self.feature_extractor, device=-1)

        # Print emotion classes
        self.emotion_classes = self.emotion_model.config.id2label
        print("Classes:", self.emotion_classes)

    def silent_inference(self, model, frame, conf, imgsz, iou, max_det):
        # Perform inference silently (without printing logs)
        return model(frame, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det)

    def analyze_emotions_img(self, image_path, output_name):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}. Please check the file path.")
            return

        results = self.detection_model(image)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = image[y1:y2, x1:x2]

                # Convert face (NumPy array) to PIL image
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                # Predict emotion using the pipeline
                emotion_prediction = self.pipe(face_pil)

                # Extract top two emotions
                emotion1 = emotion_prediction[0]['label']
                emotion2 = emotion_prediction[1]['label']
                score1 = emotion_prediction[0]['score']
                score2 = emotion_prediction[1]['score']

                # Create label
                label = f"{emotion1}({score1:.2f}) {emotion2}({score2:.2f})"
                print(label)

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the result to a file
        cv2.imwrite(f'{self.output_folder}/{output_name}', image)
        print(f"Emotion analysis saved to {self.output_folder}/{output_name}")

    def analyze_emotions_vid(self, video_path, output_name, analyse_each_x_frame, output_folder_dbfs=None):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video at {video_path}. Please check the file path.")
            return
        print(f"Movie loaded : {video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        out = cv2.VideoWriter(f'{self.output_folder}/{output_name}', fourcc, fps, (frame_width, frame_height))
        duration = total_frames / fps
        print(f"Movie duration: {duration:.1f}s @ {fps}FPS")

        # Open a CSV file to save the timeline
        timeline_file = f"{self.output_folder}/timeline_{output_name.split('.')[0]}.csv"
        with open(timeline_file, mode='w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(["Second", "Number of Detections", "Emotions"])

            frame_count = 0  # Initialize frame counter

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Process every XXth frame
                if frame_count % analyse_each_x_frame == 0:
                    second = frame_count // fps  # Calculate the current second

                    # Initial low resolution detection
                    results = self.silent_inference(self.detection_model, frame,
                                                    conf=0.3,
                                                    imgsz=self.IMG_SIZE_LOW,
                                                    iou=0.1,
                                                    max_det=self.DETECTION_THRESHOLD)

                    nb_detections_low = sum(len(r.boxes) for r in results)
                    emotions_summary = []  # Store emotions for this second

                    # High resolution detection if needed
                    if nb_detections_low >= self.DETECTION_THRESHOLD:
                        results = self.silent_inference(self.detection_model, frame,
                                                        conf=0.05,
                                                        imgsz=self.IMG_SIZE_HIGH,
                                                        iou=0.1,
                                                        max_det=600)

                        # Plot bounding boxes without emotion prediction
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for face detection only

                    # Perform emotion prediction only if nb_detections_low < DETECTION_THRESHOLD
                    else:
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                face = frame[y1:y2, x1:x2]

                                # Convert face (NumPy array) to PIL image
                                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                                # Predict emotion using the pipeline
                                emotion_prediction = self.pipe(face_pil)

                                # Extract top two emotions
                                emotion1 = emotion_prediction[0]['label']
                                emotion2 = emotion_prediction[1]['label']
                                score1 = emotion_prediction[0]['score']
                                score2 = emotion_prediction[1]['score']

                                # Create label
                                label = f"{emotion1}({score1:.2f}) {emotion2}({score2:.2f})"
                                print(label)
                                emotions_summary.append(label)

                                # Draw bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Write the timeline data for this second
                    csvwriter.writerow([second, nb_detections_low, "; ".join(emotions_summary)])

                # Write the annotated frame to the output video
                out.write(frame)

                # Increment frame count
                frame_count += 1

        # Release resources
        cap.release()
        out.release()
        print(f"Emotion analysis video saved to {self.output_folder}/{output_name}")
        print(f"Timeline saved to {timeline_file}")

        if self.azure_run :
            # Copy the output video to Azure storage
            dbutils.fs.cp(f"file:{self.output_folder}/{output_name}", f"dbfs:{output_folder_dbfs}/{output_name}")
            dbutils.fs.cp(f"file:{timeline_file}", f"dbfs:{output_folder_dbfs}/{timeline_file.split('/')[-1]}")
            print(f"Output video and timeline copied to Azure: {output_folder_dbfs}")
            
            try:
                # Initialize PostgresUtils
                postgres_utils = PostgresUtils()

                # Connect to PostgreSQL
                postgres_utils.connect()

                # Create table if it doesn't exist
                postgres_utils.create_table(table_name="videoTimeline")

                # Insert timeline data into PostgreSQL
                if os.path.exists(timeline_file):
                    postgres_utils.insert_timeline_data(table_name="videoTimeline", timeline_file=timeline_file)
                else:
                    print(f"Error: Timeline file {timeline_file} does not exist.")

            except Exception as e:
                print(f"Error during PostgreSQL operations: {e}")
            
            finally:
                # Close the connection
                postgres_utils.close_connection()

class AzureUtils:
    def __init__(self, mount_dir):
        self.mount_dir = mount_dir

    def detect_azure_run(self):
        """
        Function to detect if the code is running in an Azure environment.
        """
        args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)
        return args.get("AZURE_RUN", "false").lower() == "true"

    def mount_dir_Azure(self):
        """
        Function to mount the directory in Azure environment.
        """
        def is_mounted(mount_point):
            mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]
            return mount_point in mounts

        configs = {
            "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-application-id"),
            "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-secret-value"),
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{dbutils.secrets.get(scope='az-kv-assemblia-scope', key='sp-tenant-id')}/oauth2/token"
        }

        if not is_mounted(self.mount_dir):
            dbutils.fs.mount(
                source="abfss://data@azbstelecomparis.dfs.core.windows.net/",
                mount_point=self.mount_dir,
                extra_configs=configs
            )
            print(f"Successfully mounted {self.mount_dir}")
        else:
            print(f"{self.mount_dir} is already mounted")

    def get_latest_video(self, blob_folder):
        """
        Function to get the latest video file (by modification date) in a blob folder.
        """
        files = dbutils.fs.ls(blob_folder)
        video_files = [f for f in files if f.name.endswith('.mp4')]
        if not video_files:
            raise FileNotFoundError(f"No video files found in {blob_folder}")

        latest_file = sorted(video_files, key=lambda f: f.modificationTime, reverse=True)[0]
        print(f"Latest video file: {latest_file.path} (modified at {latest_file.modificationTime})")
        return latest_file.path

class PostgresUtils:
    def __init__(self):
        """
        Initialize the PostgreSQL connection using environment variables.
        """
        self.host = os.getenv('PGHOST')
        self.database = os.getenv('PGDATABASE')
        self.user = os.getenv('PGUSER')
        self.password = os.getenv('PGPASSWORD')
        self.port = os.getenv('PGPORT', 5432)
        self.conn = None

        print(f"PostgreSQL connection parameters: host={self.host}, database={self.database}, user={self.user}, port={self.port}")

    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                sslmode="require"
            )
            print(f"Connected to PostgreSQL database: {self.database}")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise

    def create_table(self, table_name):
        """
        Create a table for storing timeline data if it doesn't exist.
        :param table_name: Name of the table to create.
        """
        try:
            cursor = self.conn.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                second INT,
                number_of_detections INT,
                emotions TEXT
            );
            """
            cursor.execute(create_table_query)
            self.conn.commit()
            cursor.close()
            print(f"Table '{table_name}' created successfully.")
        except Exception as e:
            print(f"Error creating table '{table_name}': {e}")
            raise

    def get_last_video_id(self, table_name):
        """
        Retrieve the last videoID from the PostgreSQL table.
        :param table_name: Name of the table to query.
        :return: The last videoID or 0 if the table is empty.
        """
        try:
            cursor = self.conn.cursor()
            query = f"SELECT MAX(videoID) FROM {table_name};"
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result[0] is not None else 0
        except Exception as e:
            print(f"Error retrieving last videoID from table '{table_name}': {e}")
            raise

    def insert_timeline_data(self, table_name, timeline_file):
        """
        Insert timeline data from a CSV file into the PostgreSQL table.
        :param table_name: Name of the table to insert data into.
        :param timeline_file: Path to the timeline CSV file.
        """
        try:
            # Get the last videoID
            last_video_id = self.get_last_video_id(table_name)
            new_video_id = last_video_id + 1

            cursor = self.conn.cursor()
            with open(timeline_file, mode='r', encoding='utf-8') as csvfile:
                next(csvfile)  # Skip the header row
                for line in csvfile:
                    # Split the line into columns
                    second, number_of_detections, emotions = line.strip().split(',')
                    
                    # Insert the data into the table
                    insert_query = f"""
                    INSERT INTO {table_name} (videoID, second, number_of_detections, emotions)
                    VALUES (%s, %s, %s, %s);
                    """
                    cursor.execute(insert_query, (new_video_id, int(second), int(number_of_detections), emotions))
            self.conn.commit()
            cursor.close()
            print(f"Timeline data successfully inserted into table '{table_name}' with videoID {new_video_id}.")
        except Exception as e:
            print(f"Error inserting data into table '{table_name}': {e}")
            raise

    def close_connection(self):
        """
        Close the connection to the PostgreSQL database.
        """
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed.")

# Example usage
if __name__ == "__main__":
    # Initialize AzureUtils
    azure_utils = AzureUtils(mount_dir="/mnt/data")

    # Check if running in Azure environment
    AZURE_RUN = azure_utils.detect_azure_run()

    if AZURE_RUN:
        print("Running in Azure environment")
        
        # Mount Azure storage if needed
        azure_utils.mount_dir_Azure()

        # DBFS paths
        input_folder_images_dbfs = f"{azure_utils.mount_dir}/video/input/images"
        input_folder_video_dbfs = f"{azure_utils.mount_dir}/video/input/videos"
        output_folder_dbfs = f"{azure_utils.mount_dir}/video/output"
        yolo_model_path_dbfs = f"{azure_utils.mount_dir}/video/models/yolov8/yolov8n-face-lindevs.pt"
        emotion_model_dir_dbfs = f"{azure_utils.mount_dir}/video/models/5-HuggingFace/"

        # Get the latest video file
        video_file_dbfs = azure_utils.get_latest_video(input_folder_video_dbfs)

        # Local temp paths
        yolo_model_path_local = "/tmp/yolov8n-face-lindevs.pt"
        emotion_model_dir_local = "/tmp/emotion_model"
        video_path_local = "/tmp/video_latest.mp4"

        # Copy YOLO model locally
        if not os.path.exists(yolo_model_path_local):
            print(f"Copying YOLO model from {yolo_model_path_dbfs} → {yolo_model_path_local}")
            dbutils.fs.cp(yolo_model_path_dbfs, f"file:{yolo_model_path_local}")
        yolo_model_path = yolo_model_path_local

        # Copy emotion model locally
        if not os.path.exists(emotion_model_dir_local):
            print(f"Copying emotion model from {emotion_model_dir_dbfs} → {emotion_model_dir_local}")
            dbutils.fs.cp(emotion_model_dir_dbfs, f"file:{emotion_model_dir_local}", recurse=True)
        emotion_model_dir = emotion_model_dir_local

        # Copy the latest video file locally
        if not os.path.exists(video_path_local):
            print(f"Copying video from {video_file_dbfs} → {video_path_local}")
            dbutils.fs.cp(video_file_dbfs, f"file:{video_path_local}")
        video_path = video_path_local

        # Local output folder (create if doesn't exist)
        output_folder = "/tmp/output"
        os.makedirs(output_folder, exist_ok=True)

    else:
        print("Running in local environment")
        # Set paths for local environment
        input_folder_images = 'input/images'
        input_folder_video = 'input/videos'
        output_folder = 'output'
        yolo_model_path = 'models/yolov8/yolov8n-face-lindevs.pt'
        emotion_model_dir = 'models/5-HuggingFace/'
        video_path = 'input/videos/video_short.mp4'

    # Initialize the HumanEmotionAnalyzer with correct paths
    analyzer = HumanEmotionAnalyzer(
        input_folder_images=input_folder_images if not AZURE_RUN else input_folder_images_dbfs,
        input_folder_video=input_folder_video if not AZURE_RUN else input_folder_video_dbfs,
        output_folder=output_folder,
        yolo_model_path=yolo_model_path,
        emotion_model_dir=emotion_model_dir,
        azure_run=AZURE_RUN
    )

    # # Example for images
    # for image in os.listdir(analyzer.input_folder_images):
    #     if image.endswith('.jpeg') or image.endswith('.png'):
    #         analyzer.analyze_emotions_img(f'{analyzer.input_folder_images}/{image}', f'output_{image}')

    # Example for video
    analyzer.analyze_emotions_vid(video_path, 
                                  'output_video_test.mp4', 
                                  analyse_each_x_frame=10, # example : 1 to process every frame, 10 to process every 10th frame
                                  output_folder_dbfs=output_folder_dbfs) 