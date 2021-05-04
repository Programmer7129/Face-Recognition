"""
Important Libraries to import given below.
This makes the program a lot easier and readable to a beginner.
"""

import face_recognition as fr
import cv2

imgElon = fr.load_image_file('Images/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = fr.load_image_file('Images/Elon Test.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)  # cvt means convert

faceLoc = fr.face_locations(imgElon)[0]   # Top Right Bottom Left - 4 values for location
encodeElon = fr.face_encodings(imgElon)[0]  # Mention 0 at the end as we are generating the first value from list.
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = fr.face_locations(imgTest)[0]   # Top Right Bottom Left - 4 values for location
encodeTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = fr.compare_faces([encodeElon], encodeTest)
faceDistance = fr.face_distance([encodeElon], encodeTest)
print(results, faceDistance)

# We can use the given below line to print our result i.e, "results" and "faceDistance" on the test image.
# cv2.putText(imgTest, f'{results}{round(faceDistance[0],2)}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
