import cv2
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def stack_images(scale, imgArray,labels ):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        if len(labels) != 0:
            eachImgWidth = int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)
            # print(eachImgHeight)
            for d in range(0, rows):
                for c in range(0, cols):
                    cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 100, 200), 2)

        return ver




def find_rectangle_contours(contours):
    rectangle_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True )
            # print("corner points length", len(approx))
            if len(approx) == 4:
                rectangle_contours.append(i)
    # print(rectangle_contours)
    rectangle_contours = sorted(rectangle_contours, key = cv2.contourArea, reverse = True)
    return rectangle_contours





def get_corner_points(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return approx





def reorder(points):
    points = points.reshape((4,2))
    add = points.sum(1)                     # axis = 1   add is a list
    diff = np.diff(points, axis = 1)

    new_points = np.zeros_like(points, np.int32)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def split_boxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols  = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes


def calc_score(grading,positive,negative):
    occurances = Counter(grading)
    # left = occurances[-1]
    correct = occurances[1]
    wrong = occurances[0]
    score = (correct * positive) + (wrong * negative)
    return  score




def show_answers(img, response , grading, solution, questions, choices):
    section_width = int(img.shape[1] / questions)
    section_height = int(img.shape[0] / choices)

    for x in range(0 , questions):
        ans = response[x]
        center_x = (ans * section_width) + section_width // 2
        center_y = (x * section_height) + section_height // 2

        if grading[x] == 1:
            cv2.circle(img, (center_x, center_y), 50, (0, 255, 0), cv2.FILLED)

        elif grading[x] == 0:
            cv2.circle(img, (center_x, center_y), 50, (0, 0, 255), cv2.FILLED)
            correct_x = (solution[x] * section_width) + section_width // 2
            correct_y = (x * section_height) + section_height // 2
            cv2.circle(img, (correct_x, correct_y), 20, (255,64,25), 50)

        else:
            correct_x = (solution[x] * section_width) + section_width // 2
            correct_y = (x * section_height) + section_height // 2
            cv2.circle(img, (correct_x, correct_y), 20, (255,64,25), 50)

    return img


def splitBoxes(img,cols):

    boxes=[]
    col= np.hsplit(img,cols)
    for i in col:
        boxes.append(i)
    return boxes






def test_image(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    canny = cv2.Canny(blur,200,100)
    threshold = cv2.threshold(blur,110,255,cv2.THRESH_BINARY_INV)[1]
    print(threshold.shape)
    reshaped_img = cv2.resize(threshold,(28,28))
    img_aaray = np.array(reshaped_img)
    img_resh = img_aaray.reshape(1,28,28,1)
    pic = img_resh.astype('float32')
    pic = pic / 255.0

    cv2.imshow("demo",threshold)
    cv2.waitKey(0)




# load and prepare the image
def load_image(img):
    # # load the image
    # img = load_img(filename, grayscale=True, target_size=(28, 28))
    # # convert to array
    # img = img_to_array(img)
    # # reshape into a single sample with 1 channel
    # img = img.reshape(1, 28, 28, 1)
    # # prepare pixel data
    # img = img.astype('float32')
    # img = img / 255.0

    # img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    # canny = cv2.Canny(blur,100,100)
    threshold = cv2.threshold(blur,140,255,cv2.THRESH_BINARY_INV)[1]

    reshaped_img = cv2.resize(threshold,(28,28))
    img_aaray = np.array(reshaped_img)
    img_resh = img_aaray.reshape(1,28,28,1)
    pic = img_resh.astype('float32')
    pic = pic / 255.0
    return pic


# load an image and predict the class
def get_number(filepath):
    # load the image
    img = load_image(filepath)
    # load model
    model = load_model('model/tf_digit_model2.h5')


    # predict the class
    digit = model.predict_classes(img)
    print(digit[0])
    return digit[0]




# entry point, run the example
# run_example()
# test_image('Resources/81.jpeg')
