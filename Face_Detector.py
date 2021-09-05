import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithims)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    # Must convert to greyscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

    # Detect Faces, detectMultiscale returns coordinates face in image/video
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    # Draw rectangles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x + w , y + h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 10)
    #cv2.rectangle(img, (x,y), (x + w , y + h) , (0, 255, 0), 2)

    cv2.imshow('Sucess',frame)
    key = cv2.waitKey(1)

    # Stop if pressed q or Q
    if key == 113 or key == 81:
        break