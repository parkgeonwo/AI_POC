
# python 3.9.1
# opencv-contrib-python 4.5.5.64
# mediapipe 0.8.9.1

###################### corner detection 

# import cv2

# src = cv2.imread("./document/car_document.jpg")
# src_copy = src.copy()

# gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
# ret, image_binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
# image_binary = cv2.bitwise_not(image_binary)

# corners = cv2.goodFeaturesToTrack(image_binary, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

# for i in corners:
#     cv2.circle(image_binary, tuple( [ int(i[0][0]), int(i[0][1]) ] ), 3, (0, 0, 255), 2)


# cv2.imshow("dst", image_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




################## multi-scale hough transform

# from calendar import c
# import numpy as np
# import cv2

# src = cv2.imread("./document/car_document.jpg")
# dst = src.copy()
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)

# lines = cv2.HoughLines(image_binary, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)

# for i in lines:
#     rho, theta = i[0][0], i[0][1]
#     a, b = np.cos(theta), np.sin(theta)
#     x0, y0 = a*rho, b*rho

#     scale = src.shape[0] + src.shape[1]

#     x1 = int(x0 + scale * -b)
#     y1 = int(y0 + scale * a)
#     x2 = int(x0 - scale * -b)
#     y2 = int(y0 - scale * a)

#     cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     # cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



###################### progressive probabilistic hough transform


# import numpy as np
# import cv2

# src = cv2.imread("./document/car_document.jpg")
# dst = src.copy()
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# ret, image_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
# image_binary = cv2.bitwise_not(image_binary)
# canny = cv2.Canny(image_binary, 5000, 1500, apertureSize = 5, L2gradient = True)
# lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 10, maxLineGap = 100)

# for i in lines:
#     cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)

# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




############################# binary

# import numpy as np
# import cv2

# def empty(pos):
#     pass

# src = cv2.imread("./document/car_document.jpg", cv2.IMREAD_GRAYSCALE)
# src = cv2.resize(src, dsize=(1404, 992), interpolation=cv2.INTER_AREA)

# name = "Trackbar"
# cv2.namedWindow(name)

# cv2.createTrackbar("block_size", name, 25, 100, empty )
# cv2.createTrackbar("c", name, 3, 10, empty )

# while True:
#     block_size = cv2.getTrackbarPos("block_size", name)
#     c = cv2.getTrackbarPos("c",name)

#     if block_size <= 1:
#         block_size = 3
#     if block_size % 2 == 0:
#         block_size += 1

#     image_binary = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c )

#     cv2.imshow(name, image_binary)
#     if cv2.waitKey(1) == ord("q"):
#         break

# cv2.destroyAllWindows()



################################### drawcontours + mask

# import numpy as np
# import cv2

# src = cv2.imread("./document/car_document.jpg")
# src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# mask = np.zeros_like(src)

# ret, image_binary = cv2.threshold(src, 230, 255, cv2.THRESH_BINARY)   # 이진화

# image_binary = cv2.GaussianBlur(image_binary, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_edged = cv2.Canny(image_binary, 75,200, True)       # 엣지(외곽선) 검출, img / min 임계치 / max 임계치

# cnts, hierarchy = cv2.findContours(image_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method

# cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# cnt = cnts[0]     # contourArea가 가장 큰

# src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)    # color line 그리기 위한 조치

# cv2.drawContours(src, [cnt], -1, (255, 0, 0), 1)
# cv2.drawContours(mask, [cnt], -1, (255,0,0), -1 )

# out = np.zeros_like(src)
# out[mask==255] = src[mask==255]

# # Now crop
# (y, x) = np.where(mask == 255)
# (topy, topx) = (np.min(y), np.min(x))
# (bottomy, bottomx) = (np.max(y), np.max(x))
# out = out[topy:bottomy+1, topx:bottomx+1]

# cv2.imshow('out',out)
# cv2.imshow('image', src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


############################## findcountours

import numpy as np
import cv2

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document.jpg", cv2.IMREAD_GRAYSCALE)
# src = cv2.resize( src, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )

ret, binary_image = cv2.threshold(upload_image, 230, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
# image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
canny_image = cv2.Canny(gblur_image, 75,200, True)


cnts, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

rect = cv2.minAreaRect(cnts[0])  # largest 중 하나를 직사각형 형태로 return = (c_x,c_y) / (width, height) / angle of rotation
r = cv2.boxPoints(rect)
box = np.int0(r)

upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(upload_image, [box], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

# # 4개의 점 다른색으로 표시
boxes = [tuple(i) for i in box]
cv2.circle(upload_image, boxes[0], 1, (0, 0, 0), 5)   # 검  # boxes[0] -> x1, y1 , 좌상단
cv2.circle(upload_image, boxes[1], 1, (255, 0, 0), 5) # 파  # boxes[1] -> x2, y1 , 우상단
cv2.circle(upload_image, boxes[2], 1, (0, 255, 0), 5) # 녹  # boxes[2] -> x2, y2 , 우하단
cv2.circle(upload_image, boxes[3], 1, (0, 0, 255), 5) # 적  # boxes[3] -> x1, y2 , 좌하단

W = rect[1][1]       # 90도 돌아간거로 인식되서 rect[1][1]이 width
H = rect[1][0]

pts1 = np.float32([ [boxes[0]], [boxes[3]], [boxes[1]], [boxes[2]] ])   # 좌상, 좌하, 우상, 우하
pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

M = cv2.getPerspectiveTransform(pts1, pts2)

perspective_image = cv2.warpPerspective(binary_image, M, (int(W),int(H)))
 
# print(W, H)     # 1173, 742

left_col_x1 = int(W*0.149)          # 175 / 1173     # 왼쪽 col의 x1
left_col_x2 = int(W*0.469)         # 546 / 1173      # 왼쪽 col의 x2
col_y1 = int(H*0.171)           # 127 / 742          # 왼쪽 상단의 y1
col_y2 = int(H*0.233)               # 173 / 742      # 왼쪽 상단의 y2
row_height = col_y2 - col_y1                         # 한칸의 높이
row_interval = int(H*0.00539)                         # 칸사이의 간격

print(row_height, row_interval)

right_col_x1 = int(W*0.645)       # 756 / 1173       # 오른쪽 1번째 col의 x1
right_col_x1_1 = int(W*0.796)     # 934 / 1173       # 오른쪽 1번째 col의 x2
right_col_x1_2 = int(W*0.869)      # 1019 / 1173     # 오른쪽 2번째 col의 x1
right_col_x2 = int(W*0.995)       # 1167 / 1173      # 오른쪽 2번째 col의 x2

upper_left_x1 = int(W*0.034)      # 40 / 1173       # 표 상단 자동차 등록번호 x1
upper_right_x1 = int(W*0.739)      # 867 / 1173     # 표 상단 날짜 x1

upper_y1 = int(H*0.121)      #  90 / 742         # 표 상단 자동차 등록번호 / 날짜 y1
upper_y2 = int(H*0.162)      #  120 / 742        # 표 상단 자동차 등록번호 / 날짜 y2

cv2.imshow("perspective_image",perspective_image)

# 자동차 등록증 번호
crop_img = perspective_image[upper_y1:upper_y2, upper_left_x1:left_col_x2]
cv2.imshow("crop_img_1", crop_img)

# 날짜
crop_img = perspective_image[upper_y1:upper_y2, upper_right_x1:right_col_x2]
cv2.imshow("crop_img_2", crop_img)

for i in range(0,6):     # 자동차등록번호 -> 주소
    if i in [3,5]:
        x1, y1 = left_col_x1, col_y1 + row_height *i + row_interval *i
        x2, y2 = right_col_x2, col_y1 + row_height *(i+1) + row_interval *i
        crop_img = perspective_image[y1:y2, x1:x2]
        cv2.imshow("crop_img_%d" %(i+3), crop_img)
        # cv2.imshow("perspective_image",perspective_image)
        cv2.waitKey(0)
    else:
        x1, y1 = left_col_x1, col_y1 + row_height *i + row_interval *i
        x2, y2 = left_col_x2, col_y1 + row_height *(i+1) + row_interval *i
        crop_img = perspective_image[y1:y2, x1:x2]
        cv2.imshow("crop_img_%d" %(i+3), crop_img)
        cv2.waitKey(0)

for i in range(0,6):        # 차종 -> 주민등록번호 -> 용도
    if i == 0:
        x1, y1 = right_col_x1, col_y1 + row_height *i + row_interval *i
        x2, y2 = right_col_x1_1, col_y1 + row_height *(i+1) + row_interval *i
        crop_img = perspective_image[y1:y2, x1:x2]
        cv2.imshow("crop_img_%d" %(i+9), crop_img)
        cv2.waitKey(0)
    elif i == 1:
        x1, y1 = right_col_x1_2, col_y1 + row_height *(i-1) + row_interval *(i-1)
        x2, y2 = right_col_x2, col_y1 + row_height *(i) + row_interval *(i-1)
        crop_img = perspective_image[y1:y2, x1:x2]
        cv2.imshow("crop_img_%d" %(i+9), crop_img)
        cv2.waitKey(0)
    elif i in [2,3,5]:
        x1, y1 = right_col_x1, col_y1 + row_height *(i-1) + row_interval *(i-1)
        x2, y2 = right_col_x2, col_y1 + row_height *(i) + row_interval *(i-1)
        crop_img = perspective_image[y1:y2, x1:x2]
        if i == 5:
            cv2.imshow("crop_img_%d" %(i+8), crop_img)
            cv2.waitKey(0)
        else:
            cv2.imshow("crop_img_%d" %(i+9), crop_img)
            cv2.waitKey(0)


# cv2.imshow("image", image.upload_image)
# cv2.imshow("image2", image.binary_image)
# cv2.imshow("image3", image.canny_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


#### ocr list

# 자동차 등록증 번호 v / 최초등록일 v / 자동차등록번호  / 차종  / 용도  / 차명  / 형식 및 모델연도  / 차대번호  / 원동기형식  / 사용본거지 
# 성명(명칭)  / 주민(법인)등록번호  / 주소 

# 제원관리번호  / 길이  / 너비  / 높이  / 총중량  / 배기량  / 정격출력  / 승차정원  / 최대적재량  / 기통수  / 연료의 종류 



### 마스킹

# 주민(법인)등록번호 뒤 6자리 v


### 좌표

# left_col_x1 = int(W*0.157)
# left_col_x2 = int(W*0.528)
# left_col_y1 = int(H*0.182)
# row_height = 45
# row_interval = 4

# right_col_x1 =
# right_col_x2 = int(W*0.998)

# # 자동차등록번호
# x1, y1 = left_col_x1, left_col_y1
# x2, y2 = left_col_x2, left_col_y1 + row_height

# 차명
# x1, y1 = left_col_x1, left_col_y1 + row_height + row_interval
# x2, y2 = left_col_x2, left_col_y1 + row_height *2 + row_interval

# 차대번호
# x1, y1 = left_col_x1, left_col_y1 + row_height *2 + row_interval *2
# x2, y2 = left_col_x2, left_col_y1 + row_height *3 + row_interval *2

# # 사용본거지
# x1, y1 = left_col_x1, left_col_y1 + row_height *3 + row_interval *3
# x2, y2 = right_col_x2, left_col_y1 + row_height *4 + row_interval *3

# 성명(명칭)
# x1, y1 = left_col_x1, left_col_y1 + row_height *4 + row_interval *4
# x2, y2 = left_col_x2, left_col_y1 + row_height *5 + row_interval *4

# 주소
# x1, y1 = left_col_x1, left_col_y1 + row_height *5 + row_interval *5
# x2, y2 = right_col_x2, left_col_y1 + row_height *6 + row_interval *5


# # 차종
# x1, y1 = 744, 126
# x2, y2 = 887, 168 

# 용도
# x1, y1 = 975, 126
# x2, y2 = 1113, 168


# 형식 및 모델연도
# x1, y1 = 744, 173
# x2, y2 = 1113, 215










