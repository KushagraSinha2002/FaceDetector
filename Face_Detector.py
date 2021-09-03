import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithims)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Choose an image to stetct faces in
img = cv2.imread('rdj1.jpg')

# Must convert to greyscale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces, detectMultiscale returns coordinates face in image
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

# Draw rectangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x + w , y + h) , (randrange(128,256), randrange(128,256), randrange(128,256)), 10)
#cv2.rectangle(img, (x,y), (x + w , y + h) , (0, 255, 0), 2)


print(face_coordinates)
#[[240  91 272 272]]
# x   y   w   h

# Open the image
cv2.imshow('Sucess',img)
cv2.waitKey()

print("Code Completed")