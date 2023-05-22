import cv2
import os
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load the image
image = cv2.imread("image.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

# Draw rectangles around the detected faces
for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Create a folder to save the image
# output_folder = "detected_images"
# os.makedirs(output_folder, exist_ok=True)

# create name of images with faces
images_faces = f"image_with_faces_{time.time()}.jpg"

# Save the image with detected faces
output_path = os.path.join("detected_images", images_faces)
cv2.imwrite(output_path, image)

# Display the image with detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
