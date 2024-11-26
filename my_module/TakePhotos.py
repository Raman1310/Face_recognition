import os
import numpy as np
import pandas as pd
import cv2 as cv
from datetime import datetime

class FaceCapture:
    def __init__(self, id_names_file='ids-names.csv', faces_dir='faces', classifier_path='Classifiers/haarface.xml'):
        self.id_names_file = id_names_file
        self.faces_dir = faces_dir
        self.classifier_path = classifier_path
        self.id_names = self._load_id_names()
        self.face_classifier = cv.CascadeClassifier(self.classifier_path)

        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

    def _load_id_names(self):
        if os.path.exists(self.id_names_file):
            id_names = pd.read_csv(self.id_names_file)
            return id_names[['id', 'name']]
        else:
            id_names = pd.DataFrame(columns=['id', 'name'])
            id_names.to_csv(self.id_names_file, index=False)
            return id_names

    def register_user(self, user_id, user_name):
        if user_id not in self.id_names['id'].values:
            os.makedirs(f'{self.faces_dir}/{user_id}')
            self.id_names = self.id_names.append({'id': user_id, 'name': user_name}, ignore_index=True)
            self.id_names.to_csv(self.id_names_file, index=False)
            print(f'New user registered: {user_name} (ID: {user_id})')
        else:
            print(f'User with ID {user_id} already exists.')

    def get_user_name(self, user_id):
        if user_id in self.id_names['id'].values:
            return self.id_names[self.id_names['id'] == user_id]['name'].item()
        return None

    def capture_photos(self, user_id):
        user_name = self.get_user_name(user_id)
        if not user_name:
            print("User ID not registered. Please register first.")
            return

        print(f"Welcome {user_name}! Let's capture some photos.")
        print("Press 's' to save a photo and 'q' to quit.")

        camera = cv.VideoCapture(0)
        photos_taken = 0

        while cv.waitKey(1) & 0xFF != ord('q'):
            _, img = camera.read()
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                face_region = grey[y:y + h, x:x + w]
                if cv.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
                    face_img = cv.resize(face_region, (220, 220))
                    img_name = f'face.{user_id}.{datetime.now().microsecond}.jpeg'
                    cv.imwrite(f'{self.faces_dir}/{user_id}/{img_name}', face_img)
                    photos_taken += 1
                    print(f'{photos_taken} -> Photos taken!')

            cv.imshow('Face', img)

        camera.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    capture = FaceCapture()

    print('Welcome!')
    user_id = int(input('Enter your ID: '))

    user_name = capture.get_user_name(user_id)
    if user_name:
        print(f'Welcome Back {user_name}!!')
    else:
        user_name = input('Please Enter your name: ')
        capture.register_user(user_id, user_name)

    input("Press ENTER to start capturing photos.")
    capture.capture_photos(user_id)
