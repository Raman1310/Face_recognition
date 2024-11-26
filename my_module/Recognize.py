import subprocess
import sys
import os

# Automatically install requirements from requirements.txt
requirements_file = "requirements.txt"
def install_requirements():
    if os.path.exists(requirements_file):
        print(f"Installing packages from {requirements_file}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"{requirements_file} not found. Skipping requirements installation.")

install_requirements()

import pandas as pd
import numpy as np
import cv2 as cv
from Train import FaceTrainer
from TakePhotos import FaceCapture

class FaceRecognizer:
    def __init__(self, id_names_file='ids-names.csv', classifier_path='Classifiers/haarface.xml', model_path='Classifiers/TrainedLBPH.yml', confidence_threshold=80):
        self.id_names_file = id_names_file
        self.classifier_path = classifier_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        self.id_names = self._load_id_names()
        self.face_classifier = cv.CascadeClassifier(self.classifier_path)
        self.lbph = cv.face.LBPHFaceRecognizer_create()
        self.lbph.read(self.model_path)

    def _load_id_names(self):
        if not pd.io.common.file_exists(self.id_names_file):
            raise FileNotFoundError(f"ID names file '{self.id_names_file}' does not exist.")

        id_names = pd.read_csv(self.id_names_file)
        return id_names[['id', 'name']]

    def recognize_faces(self):
        camera = cv.VideoCapture(0)

        print("Press 'q' to quit the recognition window.")

        while cv.waitKey(1) & 0xFF != ord('q'):
            _, img = camera.read()
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            faces = self.face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

            for x, y, w, h in faces:
                face_region = grey[y:y + h, x:x + w]
                face_region = cv.resize(face_region, (220, 220))

                label, trust = self.lbph.predict(face_region)

                if trust < self.confidence_threshold:
                    name = self._get_name_by_label(label)
                else:
                    name = "Unknown"

                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(img, name, (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            cv.imshow('Recognize', img)

        camera.release()
        cv.destroyAllWindows()

    def _get_name_by_label(self, label):
        if label in self.id_names['id'].values:
            return self.id_names[self.id_names['id'] == label]['name'].item()
        return "Unknown"

if __name__ == "__main__":
    print("Welcome to the Face Recognition System")
    
    while True:
        print("\nChoose an option:")
        print("1. Capture Photos")
        print("2. Train the Model")
        print("3. Recognize Faces")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            print("\nStep 1: Capture Photos")
            capture = FaceCapture()
            user_id = int(input("Enter your ID for photo capture: "))
            user_name = capture.get_user_name(user_id)
            if user_name:
                print(f"Welcome Back {user_name}!")
            else:
                user_name = input("Enter your name: ")
                capture.register_user(user_id, user_name)
            input("Press ENTER to start capturing photos.")
            capture.capture_photos(user_id)
        
        elif choice == "2":
            print("\nStep 2: Train the Model")
            trainer = FaceTrainer()
            trainer.train_model()
        
        elif choice == "3":
            print("\nStep 3: Recognize Faces")
            recognizer = FaceRecognizer()
            recognizer.recognize_faces()
        
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")
