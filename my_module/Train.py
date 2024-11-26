import os
import pandas as pd
import numpy as np
import cv2 as cv

class FaceTrainer:
    def __init__(self, id_names_file='ids-names.csv', faces_dir='faces', model_save_path='Classifiers/TrainedLBPH.yml'):
        self.id_names_file = id_names_file
        self.faces_dir = faces_dir
        self.model_save_path = model_save_path
        self.lbph = cv.face.LBPHFaceRecognizer_create(threshold=500)

        if not os.path.exists(self.faces_dir):
            raise FileNotFoundError(f"Faces directory '{self.faces_dir}' does not exist.")

    def _load_id_names(self):
        if os.path.exists(self.id_names_file):
            id_names = pd.read_csv(self.id_names_file)
            return id_names[['id', 'name']]
        else:
            raise FileNotFoundError(f"ID names file '{self.id_names_file}' does not exist.")

    def create_training_data(self):
        faces = []
        labels = []
        for user_id in os.listdir(self.faces_dir):
            user_path = os.path.join(self.faces_dir, user_id)
            if not os.path.isdir(user_path):
                continue

            for img_name in os.listdir(user_path):
                try:
                    img_path = os.path.join(user_path, img_name)
                    face = cv.imread(img_path)
                    face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                    faces.append(face)
                    labels.append(int(user_id))
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")
        
        return np.array(faces), np.array(labels)

    def train_model(self):
        faces, labels = self.create_training_data()
        print('Training Started')
        self.lbph.train(faces, labels)
        self.lbph.save(self.model_save_path)
        print(f'Training Complete! Model saved to {self.model_save_path}')

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train_model()
