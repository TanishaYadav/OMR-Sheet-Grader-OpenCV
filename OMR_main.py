import cv2
import numpy as np
import utils
from PIL import Image as im
import base64
import io

# img_width, img_height = 700, 700
questions, choices = 5, 5
solution = [0, 2, 0, 1, 4]
positive = 2
negative = - 0.5
webcam_feed = False
count = 0

# capture = cv2.VideoCapture(0)
# capture.set(10, 150)


# while True:
#     if webcam_feed:
#         success , img = capture.read()
#     else:
#         img = cv2.imread('Resources/paper 2.jpeg')
#         img_width, img_height = 600, 750

def evaluate_result(filename, solution,positive,negative):




    img = cv2.imread(filename)
    img_width, img_height = 600, 750
    # ----------------------------------------------------- Preprocessing Phase-----------------------------------------------


    img_resized = img
    img_gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),1)
    img_canny = cv2.Canny(img_blur,10,50)
    img_final = img_resized.copy()

    # preprocessed_images_array = ([img_resized,img_gray,img_blur,img_canny])                #imageArray
    # img_stacked = utils.stack_images(0.5,preprocessed_images_array)

    try:
        # ------------------------------------------ Finding Contours and Corners------------------------------------------------

        contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        img_contours  = img_resized.copy()
        cv2.drawContours(img_contours,contours,-1,(0,235,10),5)                           # image , contours, indices(-1 for all) , color, thickness

        rect_contours = utils.find_rectangle_contours(contours)                           # all contours with 4 corners
        largest_contour = utils.get_corner_points(rect_contours[0])                # largest
        grade_contour = utils.get_corner_points(rect_contours[1])
        rollno_contour = utils.get_corner_points(rect_contours[3])

        img_biggest_contour = img_resized.copy()
        img_grade_contour = img_resized.copy()
        img_rollno_contour = img_resized.copy()

        if largest_contour.size != 0 and grade_contour.size != 0:
            cv2.drawContours(img_biggest_contour,largest_contour,-1,(0,255,0),30)
            cv2.drawContours(img_biggest_contour,grade_contour,-1,(255,255,0),30)
            cv2.drawContours(img_biggest_contour,rollno_contour,-1,(255,255,0),30)


        largest_contour = utils.reorder(largest_contour)
        grade_contour = utils.reorder(grade_contour)
        rollno_contour = utils.reorder(rollno_contour)



        # --------------------------------------------------- Transforming Perspective---------------------------------------------

        h=  600
        pt1 = np.float32(largest_contour)
        pt2 = np.float32([[0,0],[img_width,0],[0,h],[img_width,h]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        img_transformed = cv2.warpPerspective(img_resized,matrix, (img_width, h))

        pt1_grade = np.float32(grade_contour)
        pt2_grade = np.float32([[0,0],[600,0],[0,100],[600,100]])
        matrix_grade = cv2.getPerspectiveTransform(pt1_grade, pt2_grade)
        img_grade_transformed = cv2.warpPerspective(img_resized,matrix_grade, (600, 100))
        # cv2.imshow('Grade col' ,img_grade_transformed )


        pt1_rollno = np.float32(rollno_contour)
        pt2_rollno = np.float32([[0, 0], [300, 0], [0, 50], [300, 50]])
        matrix_rollno = cv2.getPerspectiveTransform(pt1_rollno, pt2_rollno)
        img_rollno_transformed = cv2.warpPerspective(img_resized, matrix_rollno, (300, 50))
        # cv2.imshow('Rollno' ,img_rollno_transformed )



        # -------------------------------------------------------- Applying threshold ---------------------------------------------

        img_transformed_gray = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
        img_threshold = cv2.threshold(img_transformed_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]           # src, threshold value, max_value, technique     intensity of white color



        # ----------------------------------------------------------Splitting Image ------------------------------------------------

        boxes = utils.split_boxes(img_threshold)

        pxl_value = np.zeros((questions, choices))
        row , col = 0 , 0


        #---------------------------------------------------- Counting Non Zero Pixel values for each box-------------------------

        for image in boxes:
            total_pxls = cv2.countNonZero(image)
            pxl_value[row][col] = total_pxls
            col = col + 1
            if col == choices:
                row = row + 1
                col = 0
        # print(pxl_value)



        #------------------------------------------------------- Getting Response -----------------------------------------------

        threshold = np.amax(pxl_value)
        response = []
        for x in range(0, questions):
            arr = pxl_value[x]                              # a row of pxl_value
            index = np.where(arr == np.amax(arr))           # np.amax(arr) --> max element

            if np.amax(arr) >= threshold * 0.7:
                response.append(index[0][0])
            else:
                response.append(-1)
        # print(response)



        # ----------------------------------------------------Reading RollNo.-----------------------------------------------------

        #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        # imgSolvedDigits = img_canvas.copy()

        boxes = utils.splitBoxes(img_rollno_transformed,4)

        # print(len(boxes))
        # cv2.imshow('box', boxes[2])
        # # cv2.imshow("Sample",boxes[65])
        # numbers = getPredection(boxes, model)
        # print(numbers)
        # imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        # numbers = np.asarray(numbers)
        # posArray = np.where(numbers > 0, 0, 1)
        # print(posArray)
        rollno_arr = []

        for i in range(len(boxes)):
            # cv2.imshow(f"digit{i}",boxes[i])
            rollno_arr.append(utils.get_number(boxes[i]))
            print(rollno_arr)
        rollno = int(''.join(str(x) for x in rollno_arr))
        print(rollno)



        #---------------------------------------------------- Evaluation of responses--------------------------------------------

        grading = []
        for x in range(0, questions):
            if solution[x] == response[x]:
                grading.append(1)
            elif response[x] == -1:
                grading.append(-1)
            else:
                grading.append(0)
        # print(grading)


        score = utils.calc_score(grading,positive, negative)
        max_score = questions * positive
        if score < 0:
            percentage = 0
        else:
            percentage = (score / max_score) * 100

        img_result = img_transformed.copy()
        img_result = utils.show_answers(img_result, response , grading, solution, questions, choices)



        # ------------------------------------ Displaying Results------------------------------------------------------------------

        img_raw_drawings = np.zeros_like(img_transformed)
        img_raw_drawings = utils.show_answers(img_raw_drawings , response , grading, solution, questions, choices)



        # --------inverse transforming----------

        inverse_matrix = cv2.getPerspectiveTransform(pt2 , pt1)
        img_inverse_transformed = cv2.warpPerspective(img_raw_drawings,inverse_matrix, (img.shape[1],img.shape[0]))


        img_raw_grade = np.zeros_like(img_grade_transformed, np.uint8)
        # cv2.putText(img_raw_grade,  str(percentage) + '%', (70 ,100),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255), 1 )

        cv2.putText(img_raw_grade, 'Score ' + str(score) +  ''' Percentage '''   + str(percentage) + '%', (10 ,50),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1 )



        matrix_inverse_grade = cv2.getPerspectiveTransform(pt2_grade, pt1_grade)
        img_grade_inverse_transformed = cv2.warpPerspective(img_raw_grade,matrix_inverse_grade, (img.shape[1],img.shape[0]))



        img_final = cv2.addWeighted(img_final,1,img_inverse_transformed,1,0)
        img_final = cv2.addWeighted(img_final,1,img_grade_inverse_transformed,1,0)

        img_canvas = np.zeros_like(img_resized)
        preprocessed_images_array = ([img_resized, img_gray, img_blur, img_canny],
                                     [img_contours, img_biggest_contour, img_transformed, img_threshold],
                                     [img_result,img_rollno_transformed, img_inverse_transformed, img_final])
    except:
        img_canvas = np.zeros_like(img_resized)
        preprocessed_images_array = ([img_resized, img_gray, img_blur, img_canny],
                                     [img_canvas, img_canvas, img_canvas, img_canvas],
                                     [img_canvas, img_canvas, img_canvas, img_canvas])

    labels = [['Original','Black and white','Blur','Edges'],
              ['Contours','Biggest Contour','Perspective','Threshold'],
              ['Result','Raw Drawing','Inverse Perspective','Final']]

    img_stacked = utils.stack_images(0.2,preprocessed_images_array,labels)
    # cv2.imshow("Preview all", img_stacked)
    # cv2.waitKey(0)
    imag =  im.fromarray(img_final)                                       # till here is the function   -- shift + tab till here and uncomment below part
    data = io.BytesIO()
    imag.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return [encoded_img_data,rollno,score,percentage,grading]
# cv2.imshow("Preview all",img_stacked)
# cv2.imshow("Grade result1", img_final)
#
# k = cv2.waitKey(0)
# if k % 256 == 27:
#     # ESC pressed
#     print("Escape hit, closing...")
#     # break
#     cv2.destroyAllWindows()
# elif k % 256 == 32:
#     # SPACE pressed
#     img_name = "opencv_frame_{}.png".format(count)
#     cv2.imwrite(img_name, img_final)
#     print("{} written!".format(img_name))
#     count += 1


# temp  = evaluate_result('Resources/paper_3.jpeg')