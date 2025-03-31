import cv2
import numpy as np
import os
import smtplib
import ssl
from email.message import EmailMessage

# Email Configuration
EMAIL_SENDER = 'vikashmp0310@gmail.com'  # Replace with your email
EMAIL_PASSWORD = 'kkzf uiae hnuw rfwb'  # Replace with your email password
EMAIL_RECEIVER = 'vikashmohan0310@gmail.com'  # Replace with recipient email

# Flag to track if an email has already been sent
email_sent = False

def send_email(image_path):
    global email_sent
    if email_sent:
        return  # Stop sending emails after the first one

    msg = EmailMessage()
    msg['Subject'] = 'Unknown Face Detected!'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content('An unknown face has been detected. See the attached image.')

    with open(image_path, 'rb') as img:
        msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='unknown.jpg')

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("Email Sent Successfully!")
    email_sent = True  # Mark that email has been sent

# Face Recognition Configuration
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'database'

print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

if not os.path.exists(datasets):
    print(f"Error: The dataset folder '{datasets}' does not exist!")
    exit()

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            labels.append(id)
            images.append(cv2.imread(path, 0))
        id += 1

(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

while True:
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        confidence = prediction[1]

        if confidence < 89:
            name = names[prediction[0]]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

            # Save the unknown face image
            unknown_face_path = "unknown.jpg"
            cv2.imwrite(unknown_face_path, im)
            print(f"Unknown face detected! Image saved as {unknown_face_path}")

            # Send email only if it has not been sent before
            send_email(unknown_face_path)

        # Draw the rectangle and label
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 3)
        cv2.putText(im, f"{name} - {confidence:.0f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)

    cv2.imshow('Face Recognition', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
