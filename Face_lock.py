import cv2
import numpy as np
from PIL import Image  # pillow package
import os

xmlpath = "C:\\Users\\prasa\\AppData\\Roaming\\Python\\Python311\\site-packages\\cv2\\data" \
          "\\haarcascade_frontalface_default.xml"

def face_sample():  # This function for Generating a sample images
    global xmlpath

    cam = cv2.VideoCapture(0,
    cv2.CAP_DSHOW)  # create a video capture object which is helpful to capture videos through webcam
    cam.set(3, 640)  # set video FrameWidth
    cam.set(4, 480)  # set video FrameHeight

    detector = cv2.CascadeClassifier(xmlpath)
    # Haar Cascade classifier is an effective object detection approach

    face_id = 1
    # Use integer ID for every new face (0,1,2,3,4,5,6,7,8,9........)

    print("Taking samples, look at camera ....... ")
    count = 0  # Initializing sampling face count

    while True:

        ret, img = cam.read()  # read the frames using the above created object
        converted_image = cv2.cvtColor(img,
        cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another
        faces = detector.detectMultiScale(converted_image, 1.3, 5)
        

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # used to draw a rectangle on any image
            count += 1

            cv2.imwrite("samples/face." + str(face_id) + '.' + str(count) + ".jpg", converted_image[y:y + h, x:x + w])
            # To capture & Save images into the datasets folder

            cv2.imshow('image', img)  # Used to display an image in a window

        k = cv2.waitKey(100) & 0xff  # Waits for a pressed key
        if k == 27:  # Press 'ESC' to stop
            break
        elif count >= 100:  # Take 100 sample (More sample --> More accuracy)
            break

    print("Samples taken now closing the program....")
    cam.release()
    cv2.destroyAllWindows()


def face_train():  # This function for Training a Model

    path = 'D:\\Angel\\samples'  # Path for samples already taken

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
    detector = cv2.CascadeClassifier(xmlpath)

    # Haar Cascade classifier is an effective object detection approach

    def Images_And_Labels(path):  # function to fetch the images and labels

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:  # to iterate particular image path

            gray_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_arr = np.array(gray_img, 'uint8')  # creating an array

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                faceSamples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("Training faces. It will take a few seconds. Wait ...")

    faces, ids = Images_And_Labels(path)    
    recognizer.train(faces, np.array(ids))

    recognizer.write('D:\\Angel\\trainer\\trainer.yml')  # Save the trained model as trainer.yml

    print("Model trained, Now we can recognize your face.")


def face_match(name='Dhaneshwar'):  # This function for Recognising the face

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
    recognizer.read('D:\\Angel\\trainer\\trainer.yml')  # load trained model
    cascadePath = xmlpath
    faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
    cam.set(3, 640)  # set video FrameWidht
    cam.set(4, 480)  # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    # flag = True

    while True:

        ret, img = cam.read()  # read the frames using the above created object

        converted_image = cv2.cvtColor(img,
        cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image

            # Check if accuracy is less them 100 ==> "0" is perfect match 
            if accuracy < 50:
                accuracy = "  {0}%".format(round(accuracy))
                cam.release()
                cv2.destroyAllWindows()
                print("Thanks for using this program, have a good day.")
                return True
            else:
                accuracy = "  {0}%".format(round(100 - accuracy))

            cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            cv2.putText(img, "Press 'Esc' for Retrain Face Lock", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27 :
            # Do a bit of cleanup
            cam.release()
            cv2.destroyAllWindows()
            return False


def reset_face():
    

    # Set the path of the folder containing the files to delete
    folder_path = "D:\\Angel\\samples"

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Loop through the list of files and delete each one
    if not files:
        print('No samples Detected')
        return
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
     
            


# reset_face()