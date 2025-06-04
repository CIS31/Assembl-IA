import os
import sys
import cv2
from ultralytics import YOLO
from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import csv  # Import CSV module for saving timeline data
import shutil  # Import shutil for file operations

class HumanEmotionAnalyzer:
    def __init__(self, input_folder_images, input_folder_video, output_folder, yolo_model_path, emotion_model_dir):
        # Configuration
        self.input_folder_images = input_folder_images
        self.input_folder_video = input_folder_video
        self.output_folder = output_folder

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

    def analyze_emotions_vid(self, video_path, output_name, analyse_each_x_frame):
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



def detect_azure_run():
    """
    Function to detect if the code is running in an Azure environment.
    This function checks for the presence of a specific widget or environment variable.
    """
    # Analyse les arguments passés au script sous forme "AZURE_RUN=true"
    args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)
    return args.get("AZURE_RUN", "false").lower() == "true"


def mount_dir_Azure(MOUNT_DIR):
    """
    Function to mount the directory in Azure environment.
    This is a placeholder function and should be implemented based on the specific Azure environment setup.
    """
    def is_mounted(mount_point):
        mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]
        return mount_point in mounts

    configs = {"fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-application-id"),
            "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-secret-value"),
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{dbutils.secrets.get(scope='az-kv-assemblia-scope', key='sp-tenant-id')}/oauth2/token"}

    if not is_mounted(MOUNT_DIR):
        dbutils.fs.mount(
        source = "abfss://data@azbstelecomparis.dfs.core.windows.net/",
        mount_point = MOUNT_DIR,
        extra_configs = configs)
        print(f"Successfully mounted {MOUNT_DIR}")
    else:
        print(f"{MOUNT_DIR} is already mounted")


# Example usage
if __name__ == "__main__":
    # Check if running in Azure environment
    AZURE_RUN = detect_azure_run()

    if AZURE_RUN:
        print("Running in Azure environment")
        
        # Mount Azure storage if needed
        MOUNT_DIR = "/mnt/data"
        mount_dir_Azure(MOUNT_DIR)

        # DBFS paths
        input_folder_images_dbfs = f"{MOUNT_DIR}/video/input/images"
        input_folder_video_dbfs = f"{MOUNT_DIR}/video/input/videos"
        output_folder_dbfs = f"{MOUNT_DIR}/video/output"
        yolo_model_path_dbfs = f"{MOUNT_DIR}/video/models/yolov8/yolov8n-face-lindevs.pt"
        emotion_model_dir_dbfs = f"{MOUNT_DIR}/video/models/5-HuggingFace/"
        video_file_dbfs = f"{input_folder_video_dbfs}/video_short.mp4"

        # Local temp paths
        yolo_model_path_local = "/tmp/yolov8n-face-lindevs.pt"
        emotion_model_dir_local = "/tmp/emotion_model"
        video_path_local = "/tmp/video_short.mp4"

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

        # Copy video file locally
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
        input_folder_images='input/images'
        input_folder_video='input/videos'
        output_folder='output'
        yolo_model_path='models/yolov8/yolov8n-face-lindevs.pt'
        emotion_model_dir='models/5-HuggingFace/'
    
    # Initialize the HumanEmotionAnalyzer with correct paths
    analyzer = HumanEmotionAnalyzer(
        input_folder_images=input_folder_images if not AZURE_RUN else input_folder_images_dbfs,
        input_folder_video=input_folder_video if not AZURE_RUN else input_folder_video_dbfs,
        output_folder=output_folder,
        yolo_model_path=yolo_model_path,
        emotion_model_dir=emotion_model_dir
    )

    # # Example for images
    # for image in os.listdir(analyzer.input_folder_images):
    #     if image.endswith('.jpeg') or image.endswith('.png'):
    #         analyzer.analyze_emotions_img(f'{analyzer.input_folder_images}/{image}', f'output_{image}')

    # Example for video
    analyzer.analyze_emotions_vid(video_path, 'output_video_test.mp4', analyse_each_x_frame=10) # example : 1 to process every frame, 10 to process every 10th frame