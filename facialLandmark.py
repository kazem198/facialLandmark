import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
webCam = True


def empty(a):
    pass


cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("blue", "BGR", 0, 255, empty)
cv2.createTrackbar("green", "BGR", 0, 255, empty)
cv2.createTrackbar("red", "BGR", 0, 255, empty)


def crateBox(img, points, scale=5):
    # print(points.shape)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    img = cv2.bitwise_and(img, mask)

    # cv2.imshow("mask", img)

    bbox = cv2.boundingRect(points)
    x, y, w, h = bbox
    imgCrop = img[y:y+h, x:x+w]
    imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
    return mask


while True:
    if webCam:
        success, img = cap.read()
    else:
        img = cv2.imread("./16897557723701.webp")

    # print(img.shape)
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(imgGRAY, 0)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),
                      color=(0, 255, 0), thickness=2)
        landMark = predictor(imgGRAY, face)
        myPoints = []

        for i in range(68):
            x = landMark.part(i).x
            y = landMark.part(i).y
            myPoints.append([x, y])

        myPoints = np.array(myPoints)
        # leftEye = crateBox(img, myPoints[36:42])
        imgLips = crateBox(img, myPoints[48:61])

        cv2.imshow("imgLips", imgLips)

        colorImgLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos("blue", "BGR")
        g = cv2.getTrackbarPos("green", "BGR")
        r = cv2.getTrackbarPos("red", "BGR")

        colorImgLips[:] = (b, g, r)
        colorImgLips = cv2.bitwise_and(imgLips, colorImgLips)
        colorImgLips = cv2.GaussianBlur(colorImgLips, (7, 7), 10)
        colorImgLips = cv2.addWeighted(img, 1, colorImgLips, .4, 1)

        cv2.imshow("BGR", colorImgLips)

    cv2.imshow("img", img)
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        cv2.destroyAllWindows()
        break
