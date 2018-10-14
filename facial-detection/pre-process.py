import os
import queue
import threading
import Object_detection

import cv2

threads = []
imageQueue = queue.Queue(60)
facePosition = queue.Queue(1)
cascPath = "haarcascade_frontalface_alt.xml"

def cameraIni(deviceNum=0, isLoopInput=False):
    global Camera
    try:
        Camera = cv2.VideoCapture(deviceNum)
        if isLoopInput:
            initLoopInput(Camera)
    except "VIDEOIO ERROR":
        pass
    # time.sleep(1)
    global faceCascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    return Camera


def getFrameFromVideo(videoPath, imgOutPath, second=1):
    video = cv2.VideoCapture(videoPath)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    numFrame = 0
    while numFrame in range(fps * second):
        if video.grab():
            flag, frame = video.retrieve()
            if not flag:
                continue
            else:
                try:
                    numFrame += 1
                    newPath = imgOutPath + str(numFrame) + ".jpg"
                    cv2.imencode('.jpg', frame)[1].tofile(newPath)
                except FileNotFoundError:
                    os.mkdir(imgOutPath)
                    cv2.imencode('.jpg', frame)[1].tofile(newPath)


def initLoopInput(cam):
    t1 = threading.Thread(target=loopInput, args=(cam,1))
    t1.start()


def loopInput(camera,num):
    isopen = True
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    f = 0
    while not camera.grab():
        pass
    while (camera.isOpened() and isopen):
        isopen, img = camera.read()
        if isopen and not (img is None):
            f += 1
            if f == fps:
                while not imageQueue.empty():
                    imageQueue.get()
                f = 0
            imageQueue.put(img)




def getImgFromCam(camNum=0, display=True, closeAfterCapture=True):  # default camera 0
    camera = cameraIni(camNum, display)
    while not camera.grab():
        pass
    success, img = camera.read()
    if not success:
        return 0
    if closeAfterCapture:
        Camera.release()
    return img

def faceDetection(img,display = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if display:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the resulting frame
        cv2.imshow("Detected",img)
        cv2.waitKey(1)
    return faces


# Test Code
# getImgFromCam(0,True,True)
# getImgFromCam(0,True,True)

# initLoopInput(cameraIni(0, True))

frame = 0

Object_detection.detect(getImgFromCam(),True)


# getFrameFromVideo("/home/luke_lu/下载/Video_Object_Detection-master/7.avi", "./tmp/")
# Camera.release()
