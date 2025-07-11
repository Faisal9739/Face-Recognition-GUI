# app.py - Main GUI Application
import sys
import os
import cv2
import numpy as np
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QProgressBar, QTextEdit, QLineEdit, QGroupBox,
    QTabWidget, QMessageBox, QListWidget
)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
INPUT_VIDEOS = os.path.join(BASE_DIR, "input_videos")
OUTPUT_VIDEOS = os.path.join(BASE_DIR, "output_videos")

# Create required directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INPUT_VIDEOS, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS, exist_ok=True)

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
                color: #FFFFFF;
            }
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 8px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #61AFEF;
            }
            QPushButton {
                background-color: #3E3E42;
                color: #DCDCDC;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 5px 10px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #4A4A4F;
            }
            QPushButton:pressed {
                background-color: #007ACC;
            }
            QLineEdit, QTextEdit, QListWidget {
                background-color: #252526;
                color: #DCDCDC;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #3F3F46;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
                width: 10px;
            }
        """)
        
        self.initUI()
        self.load_person_list()
        
    def initUI(self):
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_add_face_tab(), "Add Faces")
        tabs.addTab(self.create_train_tab(), "Train Model")
        tabs.addTab(self.create_process_tab(), "Process Video")
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Set central widget
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(tabs)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
    
    def create_add_face_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Person selection
        person_group = QGroupBox("Person Management")
        person_layout = QVBoxLayout()
        
        self.person_list = QListWidget()
        self.person_list.setMaximumHeight(150)
        
        hbox = QHBoxLayout()
        self.person_name = QLineEdit(placeholderText="Enter person name")
        add_btn = QPushButton("Add Person")
        add_btn.clicked.connect(self.add_person)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_person)
        
        hbox.addWidget(self.person_name)
        hbox.addWidget(add_btn)
        hbox.addWidget(remove_btn)
        
        person_layout.addLayout(hbox)
        person_layout.addWidget(self.person_list)
        person_group.setLayout(person_layout)
        
        # Image selection
        image_group = QGroupBox("Add Face Images")
        image_layout = QVBoxLayout()
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setStyleSheet("background-color: #1E1E1E;")
        
        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse Image")
        browse_btn.clicked.connect(self.browse_image)
        detect_btn = QPushButton("Detect & Add Faces")
        detect_btn.clicked.connect(self.detect_and_add_faces)
        
        btn_layout.addWidget(browse_btn)
        btn_layout.addWidget(detect_btn)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        
        image_layout.addWidget(self.image_preview)
        image_layout.addLayout(btn_layout)
        image_layout.addWidget(self.log_output)
        image_group.setLayout(image_layout)
        
        # Add to main layout
        layout.addWidget(person_group)
        layout.addWidget(image_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Training section
        train_group = QGroupBox("Model Training")
        train_layout = QVBoxLayout()
        
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        
        btn_layout = QHBoxLayout()
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.start_training)
        
        btn_layout.addWidget(train_btn)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        
        train_layout.addLayout(btn_layout)
        train_layout.addWidget(self.train_progress)
        train_layout.addWidget(self.train_log)
        train_group.setLayout(train_layout)
        
        # Add to main layout
        layout.addWidget(train_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_process_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Video selection
        video_group = QGroupBox("Video Processing")
        video_layout = QVBoxLayout()
        
        # Input video
        input_layout = QHBoxLayout()
        self.input_video_path = QLineEdit()
        self.input_video_path.setReadOnly(True)
        input_browse_btn = QPushButton("Browse Video")
        input_browse_btn.clicked.connect(lambda: self.browse_file("video", self.input_video_path))
        
        input_layout.addWidget(self.input_video_path)
        input_layout.addWidget(input_browse_btn)
        
        # Output video
        output_layout = QHBoxLayout()
        self.output_video_path = QLineEdit()
        self.output_video_path.setReadOnly(True)
        output_browse_btn = QPushButton("Set Output")
        output_browse_btn.clicked.connect(lambda: self.set_output_path(self.output_video_path))
        
        output_layout.addWidget(self.output_video_path)
        output_layout.addWidget(output_browse_btn)
        
        # Progress
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        
        # Buttons
        btn_layout = QHBoxLayout()
        process_btn = QPushButton("Process Video")
        process_btn.clicked.connect(self.start_processing)
        
        btn_layout.addWidget(process_btn)
        
        self.video_log = QTextEdit()
        self.video_log.setReadOnly(True)
        
        video_layout.addWidget(QLabel("Input Video:"))
        video_layout.addLayout(input_layout)
        video_layout.addWidget(QLabel("Output Video:"))
        video_layout.addLayout(output_layout)
        video_layout.addLayout(btn_layout)
        video_layout.addWidget(self.video_progress)
        video_layout.addWidget(self.video_log)
        video_group.setLayout(video_layout)
        
        # Add to main layout
        layout.addWidget(video_group)
        
        tab.setLayout(layout)
        return tab
    
    def load_person_list(self):
        self.person_list.clear()
        if os.path.exists(DATASET_DIR):
            for person in os.listdir(DATASET_DIR):
                person_path = os.path.join(DATASET_DIR, person)
                if os.path.isdir(person_path):
                    self.person_list.addItem(person)
    
    def add_person(self):
        name = self.person_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a name for the person.")
            return
        
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        self.log_output.append(f"Created folder for: {name}")
        self.load_person_list()
        self.person_name.clear()
    
    def remove_person(self):
        selected = self.person_list.currentItem()
        if not selected:
            return
            
        person = selected.text()
        reply = QMessageBox.question(
            self, "Confirm Removal", 
            f"Are you sure you want to remove {person} and all their images?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            person_dir = os.path.join(DATASET_DIR, person)
            shutil.rmtree(person_dir, ignore_errors=True)
            self.log_output.append(f"Removed: {person}")
            self.load_person_list()
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale pixmap to fit preview area
                scaled_pixmap = pixmap.scaled(
                    self.image_preview.width(), 
                    self.image_preview.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_preview.setPixmap(scaled_pixmap)
                self.log_output.append(f"Loaded image: {os.path.basename(file_path)}")
    
    def detect_and_add_faces(self):
        if not hasattr(self, 'current_image_path'):
            self.log_output.append("Please select an image first")
            return
            
        selected = self.person_list.currentItem()
        if not selected:
            self.log_output.append("Please select a person")
            return
            
        person = selected.text()
        self.log_output.append(f"Processing image for: {person}...")
        
        # Run face detection in a separate thread
        self.face_thread = FaceDetectionThread(self.current_image_path, person)
        self.face_thread.finished.connect(self.on_faces_detected)
        self.face_thread.start()
    
    def on_faces_detected(self, result):
        if result:
            self.log_output.append(f"Added {result} face(s) to dataset")
        else:
            self.log_output.append("No faces detected in the image")
    
    def start_training(self):
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            self.train_log.append("No dataset found. Please add faces first.")
            return
            
        self.train_log.append("Starting training process...")
        
        # Run training in a separate thread
        self.train_thread = TrainingThread()
        self.train_thread.progress.connect(self.train_progress.setValue)
        self.train_thread.log.connect(self.train_log.append)
        self.train_thread.finished.connect(self.on_training_complete)
        self.train_thread.start()
    
    def on_training_complete(self, success):
        if success:
            self.train_log.append("Training completed successfully!")
        else:
            self.train_log.append("Training failed. Check the logs.")
    
    def browse_file(self, file_type, target_field):
        if file_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", INPUT_VIDEOS, 
                "Video Files (*.mp4 *.avi *.mov)"
            )
            if file_path:
                target_field.setText(file_path)
    
    def set_output_path(self, target_field):
        default_name = "output.mp4"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", os.path.join(OUTPUT_VIDEOS, default_name),
            "Video Files (*.mp4)"
        )
        if file_path:
            target_field.setText(file_path)
    
    def start_processing(self):
        input_path = self.input_video_path.text()
        output_path = self.output_video_path.text()
        
        if not input_path or not os.path.exists(input_path):
            self.video_log.append("Please select a valid input video")
            return
            
        if not output_path:
            self.video_log.append("Please set an output path")
            return
            
        # Check if model exists
        model_path = os.path.join(MODEL_DIR, "face_recognizer.yml")
        if not os.path.exists(model_path):
            self.video_log.append("Model not found. Please train the model first.")
            return
            
        self.video_log.append("Starting video processing...")
        
        # Run processing in a separate thread
        self.process_thread = VideoProcessingThread(input_path, output_path)
        self.process_thread.progress.connect(self.video_progress.setValue)
        self.process_thread.log.connect(self.video_log.append)
        self.process_thread.finished.connect(self.on_processing_complete)
        self.process_thread.start()
    
    def on_processing_complete(self, success):
        if success:
            self.video_log.append("Processing completed successfully!")
        else:
            self.video_log.append("Processing failed. Check the logs.")

class FaceDetectionThread(QThread):
    finished = pyqtSignal(int)
    
    def __init__(self, image_path, person_name):
        super().__init__()
        self.image_path = image_path
        self.person_name = person_name
        self.cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
        
        # Create the cascade file if it doesn't exist
        if not os.path.exists(self.cascade_path):
            self.download_cascade()
    
    def download_cascade(self):
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, self.cascade_path)
        except Exception as e:
            print(f"Error downloading cascade: {e}")
    
    def run(self):
        # Create person directory
        person_dir = os.path.join(DATASET_DIR, self.person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Process image
        img = cv2.imread(self.image_path)
        if img is None:
            self.finished.emit(0)
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        count = 0
        for i, (x, y, w, h) in enumerate(faces):
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            output_path = os.path.join(person_dir, f"{len(os.listdir(person_dir)) + 1}.jpg")
            cv2.imwrite(output_path, face_img)
            count += 1
        
        self.finished.emit(count)

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
        self.model_path = os.path.join(MODEL_DIR, "face_recognizer.yml")
        self.id_map_path = os.path.join(MODEL_DIR, "id_mapping.txt")
        
    def run(self):
        try:
            self.log.emit("Loading dataset...")
            ids, faces, id_map = self.get_images_and_labels()
            
            if len(ids) == 0:
                self.log.emit("No faces found in dataset")
                self.finished.emit(False)
                return
                
            self.log.emit(f"Found {len(ids)} faces from {len(id_map)} persons")
            
            # Train model
            self.log.emit("Training model...")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, ids)
            recognizer.save(self.model_path)
            
            # Save ID mapping
            with open(self.id_map_path, 'w') as f:
                for id, name in id_map.items():
                    f.write(f"{id},{name}\n")
            
            self.log.emit(f"Model saved to {self.model_path}")
            self.progress.emit(100)
            self.finished.emit(True)
        except Exception as e:
            self.log.emit(f"Error during training: {str(e)}")
            self.finished.emit(False)
    
    def get_images_and_labels(self):
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        face_samples = []
        ids = []
        id_map = {}
        current_id = 0
        
        total_persons = len(os.listdir(DATASET_DIR))
        processed_persons = 0
        
        for person_name in os.listdir(DATASET_DIR):
            person_dir = os.path.join(DATASET_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            id_map[current_id] = person_name
            self.log.emit(f"Processing {person_name}...")
            
            for file in os.listdir(person_dir):
                if file.lower().endswith(("jpg", "png", "jpeg")):
                    img_path = os.path.join(person_dir, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                        
                    # Detect face
                    faces = face_cascade.detectMultiScale(img)
                    if len(faces) != 1:
                        continue
                    
                    x, y, w, h = faces[0]
                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    
                    face_samples.append(face)
                    ids.append(current_id)
            
            current_id += 1
            processed_persons += 1
            progress = int((processed_persons / total_persons) * 100)
            self.progress.emit(progress)
        
        return np.array(ids), face_samples, id_map

class VideoProcessingThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
        self.model_path = os.path.join(MODEL_DIR, "face_recognizer.yml")
        self.id_map_path = os.path.join(MODEL_DIR, "id_mapping.txt")
        
    def run(self):
        try:
            # Load resources
            face_cascade = cv2.CascadeClassifier(self.cascade_path)
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.model_path)
            
            # Load ID mapping
            id_map = {}
            if os.path.exists(self.id_map_path):
                with open(self.id_map_path, 'r') as f:
                    for line in f.readlines():
                        id, name = line.strip().split(',', 1)
                        id_map[int(id)] = name
            
            # Open video
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.log.emit(f"Error opening video: {self.input_path}")
                self.finished.emit(False)
                return
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            self.log.emit(f"Processing {total_frames} frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 3rd frame for efficiency
                frame_count += 1
                if frame_count % 3 != 0:
                    out.write(frame)
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Recognize face
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    label_id, confidence = recognizer.predict(face_roi)
                    
                    # Get name from ID
                    name = "Unknown"
                    if label_id in id_map:
                        name = id_map[label_id]
                    
                    # Draw results
                    color = (0, 255, 0) if confidence < 70 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.1f})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, color, 2)
                
                out.write(frame)
                
                # Update progress
                progress = int((frame_count / total_frames) * 100)
                self.progress.emit(progress)
                
                if frame_count % 30 == 0:
                    self.log.emit(f"Processed {frame_count}/{total_frames} frames")
            
            # Cleanup
            cap.release()
            out.release()
            self.log.emit(f"Processing complete! Output saved to {self.output_path}")
            self.finished.emit(True)
        except Exception as e:
            self.log.emit(f"Error during processing: {str(e)}")
            self.finished.emit(False)

if __name__ == "__main__":
    # Create required directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(INPUT_VIDEOS, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS, exist_ok=True)
    
    # Download Haar cascade if missing
    cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                cascade_path
            )
        except:
            print("Could not download Haar cascade. Please download it manually.")
    
    # Start application
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())