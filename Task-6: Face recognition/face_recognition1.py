import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import pywhatkit
from datetime import datetime

#MAIL
EMAIL_ADD = os.environ.get('EMAIL_ADDRESS')  
EMAIL_PASS = os.environ.get('EMAIL_PASSWORD')

def mail():

    import smtplib
    import imghdr
    from email.message import EmailMessage

    
    #Sender_Email =  'EMAIL_PASS'
  
    Reciever_Email = "nisarg4843@gmail.com" 

    #Password = EMAIL_ADD
    
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Alert Message." 
    newMessage['From'] = EMAIL_ADD                  
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('WELCOME HOME! LOOKING GOOD BACK THERE...') 

    with open('D:\\MLOps Summer\\Task 6\\faces\\Nisarg\\Face.jpg', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:

        smtp.login(EMAIL_ADD, EMAIL_PASS)              
        smtp.send_message(newMessage) 

    print('MAIL SENT SUCCESSFULLY!')

#WHATSAPP
curr_time = datetime.now()
hour = curr_time.hour
min = curr_time.minute

def whatpy():
    number = '+917698811683'
    text = 'Activity detected. Check you E-mail!'
    pywhatkit.sendwhatmsg(number, text, 19, 30)
    print('WhatsApp Message sent!')


# CREATING TRAINING DATA

# Load Haarcascade face classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    # Detecting face and returning cropped images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for(x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# Accessing webcam
cap = cv2.VideoCapture(0)
count = 0

# Collecting 100 samples of face
while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_path = 'D:\\MLOps Summer\\Task 6\\faces\\Nrupesh\\' + \
            str(count) + '.jpg'
        cv2.imwrite(file_path, face)

        cv2.putText(face, str(count), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecting Face Samples...', face)

    else:
        print('Face Not Found!')
        pass
    if cv2.waitKey(10) == 27 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print('FACE SAMPLES COLLECTED!')


# # TRAINING THE MODEL

data_path = 'D:\\MLOps Summer\\Task 6\\faces\\Nrupesh\\'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

training_Data, labels = [], []

for i, files in enumerate(onlyfiles):
    img_path = data_path + onlyfiles[i]
    images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    training_Data.append(np.asarray(images, dtype=np.int32))
    labels.append(i)

#labels = labels.np.asarray(labels, dtype=np.int32)

nisarg_model = cv2.face_LBPHFaceRecognizer.create()
nisarg_model.train(np.asarray(training_Data), np.asarray(labels))
print('Model trained...')


# DETECTING FACE

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = nisarg_model.predict(face)
        # harry_model.predict(face)

        if results[1] < 500:
            confidence = int(100 * (1 - (results[1])/400))
            display_string = str(confidence) + '%'

        cv2.putText(image, display_string, (300, 120),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (200, 0, 200), 2)

        if confidence > 90:
            cv2.putText(image, "Hey Vatsal!", (250, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Recognition', image)
            cv2.imwrite(
                'D:\\MLOps Summer\\Task 6\\faces\\Nrupesh\\Face.jpg', image)
            os.system('aws ec2 run-instances --image-id  ami-0ad704c126371a549 --count 1 --instance-type t2.micro --key-name demo1 --region ap-south-1')
            print('\n=========== Instance Launched ============\n\n')
            os.system("aws ec2 create-volume --availability-zone ap-south-1  --volume-type gp2 --size 1")
            break
        else:
            cv2.putText(image, "Hey Alien!", (250, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)

    except:
        cv2.putText(image, "Acess Denied!", (220, 120),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Looking for a User", (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognition', image)
        pass

    if cv2.waitKey(10) & 0xFF == 27:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
