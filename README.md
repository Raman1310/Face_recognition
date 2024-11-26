# Face_recognition

### About the Project

We use the OpenCV library's pre-trained models to perform the task of Face Recognition using the LBPH algorithm.
OpenCV offers several for the task of object Detection. We use the Frontal-Face Haar Cascade to detect a "face" in the frame. Once a face is detected it has a bounded box to find its location, and the face is extracted, leaving aside the other non-important details in the frame.
The LBPH algotihm is then run on the extracted face.

### Requirements

- Python 3.6+
- OpenCV
- Numpy
- Pandas

### Taking Photos
1. Run `python TakePhotos.py`
2. Enter an ID and Name
3. Press the `s` key repeatedly to take photos, once a box appears around your face. It is recommended to take atleast 25 pictures.
4. Press the `q` key when you're finished taking pictures.

### Training the Model
1. Run `python Train.py`
2. After Training is complete the program will generate the file "Classifiers/TrainedLBPH.yml"

### Recognizing
1. Run `python Recognize.py`
