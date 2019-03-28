from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import dlib
import cv2
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return((A+B) / (2*C));

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

EAR_THRESHOLD = 0.20

cam = cv2.VideoCapture(0)
time.sleep(2)

while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd];

        l_EAR = eye_aspect_ratio(leftEye)
        r_EAR = eye_aspect_ratio(rightEye)

        EAR = (l_EAR + r_EAR) / 2
		
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1);
        
        if(EAR < EAR_THRESHOLD):
        	cv2.putText(frame, "ALERT: Blinking Detected", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1);
    if key == 27:
    	break
    	
cam.release()
cv2.destroyAllWindows()

