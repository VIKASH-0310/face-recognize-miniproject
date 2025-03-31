import cv2
import os
import time

# Initialize parameters
count = 0
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'

# Get user input for the person's name
fn_name = input("Enter the Person's Name: ").strip()
path = os.path.join(fn_dir, fn_name)

# Ensure that the database directory exists
os.makedirs(path, exist_ok=True)

# Image size for face recognition
(im_width, im_height) = (112, 92)

# Load Haar cascade for face detection
if not os.path.exists(fn_haar):
    print("Error: Haar cascade file not found!")
    exit()

haar_cascade = cv2.CascadeClassifier(fn_haar)

# Open webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("-----------------------Taking pictures----------------------")
print("--------------------Give some expressions---------------------")

while count < 45:
    rval, im = webcam.read()
    if not rval:
        print("Error: Could not read frame.")
        break

    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))

    faces = haar_cascade.detectMultiScale(mini, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = sorted(faces, key=lambda x: x[3])  # Sort by height

    if faces:
        (x, y, w, h) = [v * size for v in faces[0]]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Generate next filename
        existing_files = [int(n.split('.')[0]) for n in os.listdir(path) if n.split('.')[0].isdigit()]
        pin = max(existing_files, default=0) + 1

        cv2.imwrite(os.path.join(path, f"{pin}.png"), face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        time.sleep(0.38)
        count += 1

    cv2.imshow('OpenCV', im)
    
    # Break if ESC key is pressed
    if cv2.waitKey(10) == 27:
        break

# Cleanup
print(f"{count} images taken and saved to '{fn_name}' folder in the database.")
webcam.release()
cv2.destroyAllWindows()
