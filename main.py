import cv2
import face_recognition
from facerec import facerec

cam = cv2.VideoCapture(0)
f = facerec()
f.load("images_faces/")

while True:
    ret, frame = cam.read()
    loc_arr, names_arr = f.detect(frame)
    for f_loc, n in zip(loc_arr, names_arr):
        a, b, c, d = f_loc[0], f_loc[1], f_loc[2], f_loc[3]
        cv2.putText(frame, n, (d, a), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 200), 3)
        cv2.rectangle(frame, (d, a), (b, c), (0, 0, 200), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('s'):
        break 

cam.release()
cv2.destroyAllWindows()