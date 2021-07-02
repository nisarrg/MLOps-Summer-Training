from os import read
import cv2
import numpy as np
import easyocr
import imutils


cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    try:
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        
        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            new_img = cv2.drawContours(mask, [location], 0, 255, -1)
            new_img = cv2.bitwise_and(img, img, mask=mask)

            (x,y) = np.where(mask == 255)
            (x1,y1) = (np.min(x), np.min(y))
            (x2,y2) = (np.max(x), np.max(y))
            cropped_img = gray[x1:x2+1, y1:y2+1]

            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_img)
            result

            text = result[0][-2]
            font = cv2.FONT_HERSHEY_COMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,255), thickness=2)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)

            cv2.imshow('License Plate Detector', res)
    
        else:
            cv2.putText(img, "Looking for a Number Plate", (50, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('License Plate Detector', img)

    except:
        pass

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
