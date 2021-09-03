import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('rdj1.jpg')

cv2.imshow('Sucess',img)
cv2.waitKey()

print("Code")