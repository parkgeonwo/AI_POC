import numpy as np
import cv2
import math


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i,True)      # 외각선 길이 반환
            approx = cv2.approxPolyDP(i, 0.015*peri, True)     # 외각선을 근사화(단순화) 합니다.
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


############## 이미지 전처리 (이미지 업로드, 이진화, blur, canny )

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document3.jpg", cv2.IMREAD_GRAYSCALE)
# upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
frame_height,frame_width = upload_image.shape

ret, binary_image = cv2.threshold(upload_image, 240, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
canny_image = cv2.Canny(gblur_image, 75,200, True)


############### 큰박스 3개 검출

cnts, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)


######## rect_list에 cnts를 좌상단->우하단 순서로 만들기위해서 좌표를 담아주고 정렬

n = 2

rect_list = []

for i in range(n):
    biggest = biggest_contour(cnts[i:])

    cv2.drawContours(upload_image, [biggest], -1, (255,0,0),3 )

    points = biggest.reshape(4,2)
    input_points = np.zeros((4,2), dtype = "float32")

    points_sum = points.sum(axis = 1)
    input_points[0] = points[np.argmin(points_sum)]   # top left
    input_points[3] = points[np.argmax(points_sum)]   # bottom right

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]   # top right
    input_points[2] = points[np.argmax(points_diff)]   # bottom left

    # upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)
    # cv2.line(upload_image, (int(input_points[0][0]),int(input_points[0][1])), (int(input_points[1][0]),int(input_points[1][1])),(255,0,0),5,2 )
    # cv2.line(upload_image, (int(input_points[1][0]),int(input_points[1][1])), (int(input_points[3][0]),int(input_points[3][1])),(255,0,0),5,2 )
    # cv2.line(upload_image, (int(input_points[3][0]),int(input_points[3][1])), (int(input_points[2][0]),int(input_points[2][1])),(255,0,0),5,2 )
    # cv2.line(upload_image, (int(input_points[2][0]),int(input_points[2][1])), (int(input_points[0][0]),int(input_points[0][1])),(255,0,0),5,2 )

    rect_list.append(input_points)

rect_list2 = []
rect_list3 = []

height_list = [0, frame_height*0.5, frame_height ]

for j in range(len(height_list)-1):
    rect_list2 = [rect_list[i] for i in range(len(rect_list)) if ((rect_list[i][0][1]+rect_list[i][3][1])/2) >= height_list[j] and ((rect_list[i][0][1]+rect_list[i][3][1])/2) <= height_list[j+1] ]
    rect_list2.sort(key=lambda x: x[0][0])
    for i in rect_list2:
        rect_list3.append(i)

################## 각 좌표에따라 원근변환

for k in range(n):
    input_points2 = rect_list3[k]
    
    (top_left, top_right, bottom_left, bottom_right) = input_points2
    bottom_width = np.sqrt( ((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2) )
    top_width = np.sqrt( ((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2) )
    right_height = np.sqrt( ((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2) )
    left_height = np.sqrt( ((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2) )

    max_width = max( int(bottom_width), int(top_width) )
    max_height = max( int(right_height), int(left_height) )

    converted_points = np.float32( [[0,0], [max_width,0], [0,max_height], [max_width, max_height]] )

    matrix = cv2.getPerspectiveTransform(input_points2, converted_points)
    crop_image = cv2.warpPerspective( upload_image , matrix, (max_width,max_height))


    cv2.imshow("upload_image", upload_image)
    cv2.imshow("crop_image", crop_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()










    ##################### 큰박스안에서 작은 박스찾기

    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    ret, small_binary_image = cv2.threshold(crop_image, 240, 255, cv2.THRESH_BINARY)   # 이진화


    small_cnts, hierarchy = cv2.findContours(small_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
    small_cnts = sorted(small_cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적
    # print(len(small_cnts))

    ##################### 작은박스들 좌상단 순서로 정렬

    small_rect_list = []    # rect를 담기위한 list

    for i in range(len(small_cnts)):
        small_rect_list.append(cv2.minAreaRect(small_cnts[i]))   # small_rect_list에 검출된 rect 수만큼 추가

    small_rect_list2 = []
    small_rect_list3 = []


    if k == 0 :
        height_list = [0, max_height*0.1617, max_height*0.2318, max_height*0.3073, max_height*0.3733, max_height*0.442, max_height*0.5054, max_height*0.5701, max_height ]
        # 120, 172, 228, 277, 328, 375, 423, 734    / 742, 1173 = height, width

    elif k == 1:
        height_list = [0,max_height*0.0513, max_height*0.1099, max_height*0.1526, max_height*0.1954,max_height*0.2357,max_height*0.2772,max_height*0.3541,max_height*0.419,max_height ]
        # 0, 42, 90, 125, 160, 193, 227, 290, 434  / 819, 566
    elif k == 2:
        break

    for j in range(len(height_list)):
        small_rect_list2 = [small_rect_list[i] for i in range(len(small_rect_list)) if small_rect_list[i][0][1] >= height_list[j] and small_rect_list[i][0][1] <= height_list[j+1] ]
        small_rect_list2.sort(key=lambda tup: tup[0][0])
        for i in small_rect_list2:
            small_rect_list3.append(i)



    #################### m번 박스 추출

    image_number = 0

    for m in range(len(small_rect_list3)):

        rect = small_rect_list3[m]
        # print(rect)
        r = cv2.boxPoints(rect)
        copy_r = r.copy()

        if rect[2] > 10:        # 각도가 10도이상이라면, 좌표값 순서를 통일하기 위해 변경
            r[0], r[1], r[2], r[3] = copy_r[3],copy_r[0],copy_r[1],copy_r[2] 

        box2 = np.int0(r)     # r을 int로 변환
        

        if rect[2] > 10:
            W = rect[1][1]
            H = rect[1][0]
        else:
            W = rect[1][0]
            H = rect[1][1]

        if W > 30 and H >30:

            crop_image2 = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(crop_image2, [box2], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기
            cv2.imshow("crop_image2", crop_image2)

            OCR_crop_image = crop_image2[ box2[1][1] : box2[1][1] + int(H) , box2[1][0] : box2[1][0] + int(W) ]
            cv2.imshow("OCR_crop_image_{}".format(image_number), OCR_crop_image)
            # cv2.imwrite("./example/OCR_crop_image_{}.png".format(image_number), OCR_crop_image)
            image_number += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# cv2.waitKey(0)
# cv2.destroyAllWindows()






