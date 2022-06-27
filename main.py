# from Car_document_ocr import OCR

# car_ocr = OCR( upload_image_path="./document/car_document_0621_6.jpg" )

# car_ocr.crop_image_save(save_path = "./crop_image/",show_image=True)

# data =  car_ocr.ocr_data_save(image_path = "./crop_image/", ocr_type="tesseract") 
# print(data)

# df = car_ocr.save_csv(data,save_csv=True)



##############################################################3



# import numpy as np
# import cv2


# upload_image = cv2.imread("/home/matrix-5/Desktop/code/AI_POC/document/car_document_0621_7.jpg", cv2.IMREAD_GRAYSCALE)
# # upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
# frame_height,frame_width = upload_image.shape

# ret, binary_image = cv2.threshold(upload_image, 230, 255, cv2.THRESH_BINARY)   # 이진화

# gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# canny_image = cv2.Canny(gblur_image, 75,200, True)


# ############### 큰박스 3개 검출


# cnts, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# # upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)
# print(cnts)

# cv2.imshow("image", binary_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()




###########################################




# import cv2

# src = cv2.imread("/home/matrix-5/Desktop/code/AI_POC/document/car_document_0621_6.jpg", cv2.IMREAD_COLOR)


# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
# binary = cv2.bitwise_not(binary)

# cv2.imshow("src", binary) 
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# for i in range(3):
#     cv2.drawContours(src, [cnts[i]], 0, (0, 0, 255), 2)
#     cv2.putText(src, str(i), tuple(cnts[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
#     print(i, hierarchy[0][i])
#     cv2.imshow("src", src)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()




# #############################################################3

import cv2
import numpy as np


## 클래스에서 지정해줘야하는 변수들

upload_image_path="./document/car_document.jpg"
save_path = "./crop_image/"
show_image = True
image_path = "./crop_image/"
ocr_type="easyocr"
save_csv=True


############## 이미지 전처리 과정 (이미지 업로드, 이진화, blur, canny ) + 필요한 값 지정

# 이미지 업로드
upload_image = cv2.imread(upload_image_path)
# 업로드된 이미지 heigth, width
upload_image_height,upload_image_width, c = upload_image.shape

# size를 1240,1755로 맞춤
if upload_image_height < 1000 or upload_image_width < 1000: 
    upload_image = cv2.resize(upload_image, dsize=(1240, 1755), interpolation=cv2.INTER_LANCZOS4)

# 업로드 이미지가 작을때 1.5배 확대
# self.extend_size = 1
# if self.upload_image_height < 1000 or self.upload_image_width < 1000:     
#     self.extend_size = 1.57
    
#     self.upload_image = cv2.resize( self.upload_image, None, fx = self.extend_size, fy = self.extend_size, interpolation = cv2.INTER_LANCZOS4 )
#     # upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )

# BGR2GRAY
upload_image_gray = cv2.cvtColor(upload_image, cv2.COLOR_BGR2GRAY)
# gray image height, width
frame_height,frame_width = upload_image_gray.shape

# binary
ret, binary_image = cv2.threshold(upload_image_gray, 230, 255, cv2.THRESH_BINARY)   # 이진화

# blur
gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# canny
canny_image = cv2.Canny(gblur_image, 75,200, True)

# 가장 큰 box 몇개 추출할지 지정
detect_box_num = 3
# box마다 중심점 높이의 범위를 지정해주는 list
box_center_list = [[0, 0.5, 1],    # 가장 큰 box 3개
            [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],    # 가장 큰 박스 3개중 1번째 박스의 높이 리스트
            [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],   # 가장 큰 박스 3개중 2번째 박스의 높이 리스트
            None ]  # 가장 큰 박스 3개중 3번째 박스의 높이 리스트 

# 사이즈 순으로 box추출하기 위한 함수
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


# cv2.imshow("image",upload_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 저장 경로 / image를 바로 열어서 볼것인지 지정

processed_image = canny_image

############### 큰박스 detect_box_num개 검출
cnts, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# print(cnts)

######## rect_list에 cnts를 좌상단->우하단 순서로 만들기위해서 좌표를 담아주고 정렬

rect_list = []

for i in range(detect_box_num):
    contours = cnts[i:]
    biggest = biggest_contour(contours)
    # print(biggest)

    # biggest = cnts[i]
    # print(biggest)

    cv2.drawContours(upload_image, [biggest], -1, (255,0,0),3 )

    cv2.imshow("upload_image", upload_image)
    cv2.waitKey(0)

    # box의 4개의 꼭지점 추출해서 rect_list에 담음
    points = biggest.reshape(4,2)
    print(points)
    input_points = np.zeros((4,2), dtype = "float32")

    points_sum = points.sum(axis = 1)
    input_points[0] = points[np.argmin(points_sum)]   # top left
    input_points[3] = points[np.argmax(points_sum)]   # bottom right

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]   # top right
    input_points[2] = points[np.argmax(points_diff)]   # bottom left

    rect_list.append(input_points)

    rect_list2 = []
    rect_list3 = []

frame_height, frame_width = processed_image.shape

cv2.destroyAllWindows()
















