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

# src = cv2.imread("/home/matrix-5/Desktop/code/AI_POC/document/car_document.jpg")
# src_height,src_width, c = src.shape     # 이미지 사이즈 
# # if src_height < 1000 or src_width < 1000: 
# #     src = cv2.resize(src, dsize=(1240, 1755), interpolation=cv2.INTER_LANCZOS4)

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)  # gray scale 변형
# # ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# # binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
# # binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
# # binary = cv2.bitwise_not(binary)

# blur = cv2.GaussianBlur(gray,(5,5),0)
# # canny_image = cv2.Canny(gblur_image, 75,200, True)

# ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# cv2.imshow("src", binary)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# # cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# # for i in range(3):
# #     cv2.drawContours(src, [cnts[i]], 0, (0, 0, 255), 2)
# #     cv2.putText(src, str(i), tuple(cnts[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
# #     print(i, hierarchy[0][i])
# #     cv2.imshow("src", src)
# #     cv2.waitKey(0)

# # cv2.destroyAllWindows()

#################################################################3




# ################################################################# 작은 박스들도 원근변환 시키는 방법

# import cv2
# import numpy as np
# import math

# class OCR():
#     def __init__(self,upload_image_path):
#         self.img = cv2.imread(upload_image_path)

#         # 가장 큰 box 몇개 추출할지 지정
#         self.detect_large_box_num = 3

#         # box마다 중심점 높이의 범위를 지정해주는 list
#         self.box_center_list = [[0, 0.5, 1],    # 가장 큰 box 3개
#             [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],    # 가장 큰 박스 3개중 1번째 박스의 높이 리스트
#             [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],   # 가장 큰 박스 3개중 2번째 박스의 높이 리스트
#             None ]  # 가장 큰 박스 3개중 3번째 박스의 높이 리스트 

#         self.save_path = "./crop_image/"


#     def process(self):
#         img_height,img_width, c = self.img.shape
#         gray_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
#         gblur_img = cv2.GaussianBlur(gray_img,(5,5),0)
#         canny_img = cv2.Canny(gblur_img, 75,200, True)
#         ret, binary = cv2.threshold(canny_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#         large_box_cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if len(large_box_cnts) < self.detect_large_box_num:      # box 검출 수가 적으면
#             errer_message = "len(large_box_cnts) is too small"
#             print(errer_message)

#         large_box_cnts = sorted(large_box_cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적 # 면적크기에 따라 정렬

#         # for i in range(self.detect_large_box_num):    # large_box_cnts 크기별로 정렬한 box 3위까지 그려주기
#         #     cv2.drawContours(self.img, [large_box_cnts[i]], 0, (0, 0, 255), 2)
#         #     cv2.putText(self.img, str(i), tuple(large_box_cnts[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

#         large_box_list = []    # 큰 box 3개 좌표 담을 list

#         for i in range(self.detect_large_box_num):
#             reshape_size_x, reshape_size_y = large_box_cnts[i].shape[0],large_box_cnts[i].shape[2]   # reshape할 size 정의
#             points = large_box_cnts[i].reshape(reshape_size_x, reshape_size_y)             # reshape
#             large_input_points = np.zeros((4,2), dtype = "float32")                    # 틀 만들어 놓기

#             points_sum = points.sum(axis = 1)
#             large_input_points[0] = points[np.argmin(points_sum)]   # top left
#             large_input_points[3] = points[np.argmax(points_sum)]   # bottom right

#             points_diff = np.diff(points, axis=1)
#             large_input_points[1] = points[np.argmin(points_diff)]   # top right
#             large_input_points[2] = points[np.argmax(points_diff)]   # bottom left

#             large_box_list.append(large_input_points)

#         large_box_list2 = []
#         large_box_list3 = []

#         # 큰 box들의 중심 좌표를 구분할 높이 리스트
#         large_box_height_list = [ i*img_height for i in self.box_center_list[0] ]

#         # 같은 높이 범위에 있는 box들을 large_box_list2에 담고 좌측->우측으로 정렬한다음 
#         # 들어있는 box들을 large_box_list3에 append
#         for j in range(len(large_box_height_list)-1):
#             large_box_list2 = [large_box_list[i] for i in range(len(large_box_list))
#                     if ((large_box_list[i][0][1]+large_box_list[i][3][1])/2) >= large_box_height_list[j] and ((large_box_list[i][0][1]+large_box_list[i][3][1])/2) <= large_box_height_list[j+1] ]
#             large_box_list2.sort(key=lambda x: x[0][0])
#             for i in large_box_list2:
#                 large_box_list3.append(i)

#         ############ 추출된 큰 box를 하나씩 원근변환하고 큰box안에서 작은box들을 crop하여 저장

#         for k in range(self.detect_large_box_num):
            
#             ############## k번째 큰box 원근변환
#             large_input_points2 = large_box_list3[k]

#             (top_left, top_right, bottom_left, bottom_right) = large_input_points2
#             bottom_width = np.sqrt( ((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2) )
#             top_width = np.sqrt( ((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2) )
#             right_height = np.sqrt( ((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2) )
#             left_height = np.sqrt( ((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2) )

#             crop_max_width = max( int(bottom_width), int(top_width) )
#             crop_max_height = max( int(right_height), int(left_height) )

#             converted_points = np.float32( [[0,0], [crop_max_width,0], [0,crop_max_height], [crop_max_width, crop_max_height]] )

#             matrix = cv2.getPerspectiveTransform(large_input_points2, converted_points)
#             crop_img = cv2.warpPerspective( self.img , matrix, (crop_max_width,crop_max_height))

#             ################# k번째 큰박스(crop_image)안에서 작은 박스찾기
#             crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
#             crop_gblur = cv2.GaussianBlur(crop_img,(5,5),0)
#             ret, crop_binary = cv2.threshold(crop_gblur, 250,255, cv2.THRESH_BINARY)
#             # ret, crop_binary = cv2.threshold(crop_gblur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             crop_canny = cv2.Canny(crop_binary, 0,20, True)
#             crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
            
#             small_box_cnts, hierarchy = cv2.findContours(crop_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             small_box_cnts = sorted(small_box_cnts, key=cv2.contourArea, reverse=True) 


#             small_box_list = []

#             for i in range(len(small_box_cnts)):
#                 reshape_size_x, reshape_size_y = small_box_cnts[i].shape[0],small_box_cnts[i].shape[2]   # reshape할 size 정의
#                 points = small_box_cnts[i].reshape(reshape_size_x, reshape_size_y)             # reshape
#                 small_input_points = np.zeros((4,2), dtype = "float32")                    # 틀 만들어 놓기

#                 points_sum = points.sum(axis = 1)
#                 small_input_points[0] = points[np.argmin(points_sum)]   # top left
#                 small_input_points[3] = points[np.argmax(points_sum)]   # bottom right

#                 points_diff = np.diff(points, axis=1)
#                 small_input_points[1] = points[np.argmin(points_diff)]   # top right
#                 small_input_points[2] = points[np.argmax(points_diff)]   # bottom left

#                 small_box_list.append(small_input_points)


#             for i in range(self.detect_large_box_num):
#                 if self.box_center_list[k+1] == None:
#                     small_box_height_list = [0,0.5,1]
#                 elif k == i:
#                     small_box_height_list = [ j*crop_max_height for j in self.box_center_list[k+1] ]

#             small_box_list2 = []
#             small_box_list3 = []

#             for j in range(len(small_box_height_list)-1):
#                 small_box_list2 = [small_box_list[i] for i in range(len(small_box_list)) if small_box_list[i][0][1] >= small_box_height_list[j] and small_box_list[i][0][1] <= small_box_height_list[j+1] ]
#                 small_box_list2.sort(key=lambda tup: tup[0][0])
#                 for i in small_box_list2:
#                     small_box_list3.append(i)

#             #################### m번 박스 추출

#             image_num = 0
#             for m in range(len(small_box_list3)):

#                 small_crop_box = small_box_list3[m]
#                 small_crop_box_width = small_crop_box[1][0] - small_crop_box[0][0]
#                 small_crop_box_height = small_crop_box[2][1] - small_crop_box[0][1]

#                 if small_crop_box_width > (crop_max_width*3/100) and small_crop_box_height > (crop_max_height*3/100):   # crop size작은애들 걸러주기
#                     ############## k번째 큰box 원근변환
#                     small_input_points2 = small_crop_box

#                     (top_left, top_right, bottom_left, bottom_right) = small_input_points2
#                     bottom_width = np.sqrt( ((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2) )
#                     top_width = np.sqrt( ((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2) )
#                     right_height = np.sqrt( ((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2) )
#                     left_height = np.sqrt( ((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2) )

#                     small_crop_max_width = max( int(bottom_width), int(top_width) )
#                     small_crop_max_height = max( int(right_height), int(left_height) )

#                     converted_points = np.float32( [[0,0], [small_crop_max_width,0], [0,small_crop_max_height], [small_crop_max_width, small_crop_max_height]] )

#                     matrix = cv2.getPerspectiveTransform(small_input_points2, converted_points)
#                     ocr_crop_img = cv2.warpPerspective( crop_img , matrix, (small_crop_max_width,small_crop_max_height))

#                     cv2.imshow("crop_img", crop_img)
#                     cv2.imshow("OCR_crop_image_{}_{}".format(k,image_num), ocr_crop_img)
#                     image_num += 1
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()


#####################################################################################################3


# #############################################################3

# import cv2
# import numpy as np


# ## 클래스에서 지정해줘야하는 변수들

# upload_image_path="./document/car_document.jpg"
# save_path = "./crop_image/"
# show_image = True
# image_path = "./crop_image/"
# ocr_type="easyocr"
# save_csv=True


# ############## 이미지 전처리 과정 (이미지 업로드, 이진화, blur, canny ) + 필요한 값 지정

# # 이미지 업로드
# upload_image = cv2.imread(upload_image_path)
# # 업로드된 이미지 heigth, width
# upload_image_height,upload_image_width, c = upload_image.shape

# # size를 1240,1755로 맞춤
# if upload_image_height < 1000 or upload_image_width < 1000: 
#     upload_image = cv2.resize(upload_image, dsize=(1240, 1755), interpolation=cv2.INTER_LANCZOS4)

# # 업로드 이미지가 작을때 1.5배 확대
# # self.extend_size = 1
# # if self.upload_image_height < 1000 or self.upload_image_width < 1000:     
# #     self.extend_size = 1.5
# #     self.upload_image = cv2.resize( self.upload_image, None, fx = self.extend_size, fy = self.extend_size, interpolation = cv2.INTER_LANCZOS4 )
# #     # upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )

# # BGR2GRAY
# upload_image_gray = cv2.cvtColor(upload_image, cv2.COLOR_BGR2GRAY)
# # gray image height, width
# frame_height,frame_width = upload_image_gray.shape

# # binary
# ret, binary_image = cv2.threshold(upload_image_gray, 230, 255, cv2.THRESH_BINARY)   # 이진화

# # blur
# gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# # canny
# canny_image = cv2.Canny(gblur_image, 75,200, True)

# # 가장 큰 box 몇개 추출할지 지정
# detect_box_num = 3
# # box마다 중심점 높이의 범위를 지정해주는 list
# box_center_list = [[0, 0.5, 1],    # 가장 큰 box 3개
#             [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],    # 가장 큰 박스 3개중 1번째 박스의 높이 리스트
#             [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],   # 가장 큰 박스 3개중 2번째 박스의 높이 리스트
#             None ]  # 가장 큰 박스 3개중 3번째 박스의 높이 리스트 

# # 사이즈 순으로 box추출하기 위한 함수
# def biggest_contour(contours):
#     biggest = np.array([])
#     max_area = 0
#     for i in contours:
#         area = cv2.contourArea(i)
#         if area > 100:
#             peri = cv2.arcLength(i,True)      # 외각선 길이 반환
#             approx = cv2.approxPolyDP(i, 0.015*peri, True)     # 외각선을 근사화(단순화) 합니다.
#             if area > max_area and len(approx) == 4:
#                 biggest = approx
#                 max_area = area
#     return biggest


# # cv2.imshow("image",upload_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # 저장 경로 / image를 바로 열어서 볼것인지 지정

# processed_image = canny_image

# ############### 큰박스 detect_box_num개 검출
# cnts, hierarchy = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# # print(cnts)

# ######## rect_list에 cnts를 좌상단->우하단 순서로 만들기위해서 좌표를 담아주고 정렬

# rect_list = []

# for i in range(detect_box_num):
#     # contours = cnts[i:]
#     # biggest = biggest_contour(contours)
#     # print(biggest.shape)

#     biggest = cnts[i]
#     print(biggest.shape)

#     cv2.drawContours(upload_image, [biggest], -1, (255,0,0),3 )

#     cv2.imshow("upload_image", upload_image)
#     cv2.waitKey(0)

#     # box의 4개의 꼭지점 추출해서 rect_list에 담음
#     points = biggest.reshape(4,2)
#     # print(points)
#     input_points = np.zeros((4,2), dtype = "float32")

#     points_sum = points.sum(axis = 1)
#     input_points[0] = points[np.argmin(points_sum)]   # top left
#     input_points[3] = points[np.argmax(points_sum)]   # bottom right

#     points_diff = np.diff(points, axis=1)
#     input_points[1] = points[np.argmin(points_diff)]   # top right
#     input_points[2] = points[np.argmax(points_diff)]   # bottom left

#     rect_list.append(input_points)

#     rect_list2 = []
#     rect_list3 = []

# frame_height, frame_width = processed_image.shape

# cv2.destroyAllWindows()







