import numpy as np
import cv2
import math

### 이미지 전처리

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document3.jpg", cv2.IMREAD_GRAYSCALE)
# upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
frame_height,frame_width = upload_image.shape

ret, binary_image = cv2.threshold(upload_image, 240, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
# image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
canny_image = cv2.Canny(gblur_image, 75,200, True)



######## 큰박스 3개 검출

cnts, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적
cnt = cnts[:3]  # 가장 큰 BOX 3개만 검출


####### 큰 박스 3개 좌상단 순서대로 정렬

rect_list = []    # rect를 담기위한 list

for i in range(len(cnt)):      # cnts -1 수만큼 (가장 마지막꺼 제외) 
    rect_list.append(cv2.minAreaRect(cnt[i]))   # rect_list에 검출된 rect 수만큼 추가

rect_list2 = []
rect_list3 = []

height_list = [0, frame_height*0.5, frame_height ]

for j in range(len(height_list)):
    rect_list2 = [rect_list[i] for i in range(len(rect_list)) if rect_list[i][0][1] >= height_list[j] and rect_list[i][0][1] <= height_list[j+1] ]
    rect_list2.sort(key=lambda tup: tup[0][0])
    for i in rect_list2:
        rect_list3.append(i)


###### n번 box 추출 및 원근변환

n = 0

rect = rect_list3[n]
r = cv2.boxPoints(rect)
copy_r = r.copy()

if rect[2] > 10:        # 각도가 10도이상이라면 
    r[0], r[1], r[2], r[3] = copy_r[3],copy_r[0],copy_r[1],copy_r[2] # 좌표값 순서를 통일하기 위해 변경

extend_box = np.array([[7,-7],[7,7],[-7,7],[-7,-7]])

box = np.int0(r) + extend_box

upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(upload_image, [box], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

# # 4개의 점 다른색으로 표시
boxes = [tuple(i) for i in box]
cv2.circle(upload_image, boxes[0], 1, (0, 0, 0), 5)   # 검  # boxes[0] -> x1, y1 , 좌하단
cv2.circle(upload_image, boxes[1], 1, (255, 0, 0), 5) # 파  # boxes[1] -> x2, y1 , 좌상단
cv2.circle(upload_image, boxes[2], 1, (0, 255, 0), 5) # 녹  # boxes[2] -> x2, y2 , 우상단
cv2.circle(upload_image, boxes[3], 1, (0, 0, 255), 5) # 적  # boxes[3] -> x1, y2 , 우하단

if rect[2] > 10:
    W = rect[1][1]
    H = rect[1][0]
else:
    W = rect[1][0]
    H = rect[1][1]

pts1 = np.float32([ [boxes[1]], [boxes[0]], [boxes[2]], [boxes[3]] ])   # 좌상, 좌하, 우상, 우하
pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

M = cv2.getPerspectiveTransform(pts1, pts2)

# perspective_image = cv2.warpPerspective(binary_image, M, (int(W),int(H)))



######## crop 한 이미지 회전시켜서 보여주기
# def getSubImage(rect, src):
#     center, temp_size, theta = rect   # Get center, size, and angle from rect 
#     if theta < 10:
#         size = temp_size
#     else:
#         size = (temp_size[1], temp_size[0])

#     center, size = tuple(map(int, center)), tuple(map(int, size))     # Convert to int
#     M = cv2.getRotationMatrix2D( center, theta, 1)    # 사각형 회전하기 위한 정보 추출, center/반시계방향 회전각도/추가적인 확대비율
#     dst = cv2.warpAffine(src, M, (0,0) )       # Affine 이동변환(회전) , img/2x3어파인변환행렬/결과영상크기
#     out = cv2.getRectSubPix(dst, size, center)        # frame/(w,h)/(c_x,c_y)
#     return out

# if rect[2] < 10:      # 각도가 10도보다 작으면
#     crop_image = getSubImage(rect, upload_image)      # 회전된 각도만큼 다시 회전
#     binary_crop_image = getSubImage(rect, binary_image)
#     cv2.imshow('crop_image', crop_image)
#     cv2.imshow('binary_crop_image', binary_crop_image)
# else:      # 각도가 10도 이상이라면
#     crop_image = cv2.warpPerspective(upload_image, M, (int(W),int(H)))     # 원근변환
#     binary_crop_image = cv2.warpPerspective(binary_image, M, (int(W),int(H)))
#     cv2.imshow("crop_image", crop_image)
#     cv2.imshow("binary_crop_image", binary_crop_image)


crop_image = cv2.warpPerspective(upload_image, M, (int(W),int(H)))     # 원근변환
binary_crop_image = cv2.warpPerspective(binary_image, M, (int(W),int(H)))
cv2.imshow("crop_image", crop_image)
cv2.imshow("binary_crop_image", binary_crop_image)

cv2.imshow("upload_image",upload_image)









# ###################### 큰박스안에서 작은 박스찾기

cnts, hierarchy = cv2.findContours(binary_crop_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적


print(len(cnts))          # 26개의 box가 검출되어야한다.

# rect_list = []    # rect를 담기위한 list

# for i in range(len(cnts)-1):      # cnts -1 수만큼 (가장 마지막꺼 제외) 
#     rect_list.append(cv2.minAreaRect(cnts[i]))   # rect_list에 검출된 rect 수만큼 추가

# rect_list2 = []
# rect_list3 = []

# height_list = [0, H*0.1617, H*0.2318, H*0.3073, H*0.3733, H*0.442, H*0.5054, H*0.5701, H ]
# # 120, 172, 228, 277, 328, 375, 423, 734

# for j in range(len(height_list)):
#     rect_list2 = [rect_list[i] for i in range(len(rect_list)) if rect_list[i][0][1] >= height_list[j] and rect_list[i][0][1] <= height_list[j+1] ]
#     rect_list2.sort(key=lambda tup: tup[0][0])
#     for i in rect_list2:
#         rect_list3.append(i)

# print(len(rect_list3))

# rect = rect_list3[0]
# r = cv2.boxPoints(rect)
# copy_r = r.copy()

# if rect[2] > 10:        # 각도가 10도이상이라면, 좌표값 순서를 통일하기 위해 변경
#     r[0], r[1], r[2], r[3] = copy_r[3],copy_r[0],copy_r[1],copy_r[2] 

# box = np.int0(r)     # r을 int로 변환

# binary_crop_image = cv2.cvtColor(binary_crop_image, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(binary_crop_image, [box], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

# # # 4개의 점 다른색으로 표시
# boxes = [tuple(i) for i in box]
# cv2.circle(binary_crop_image, boxes[0], 1, (0, 0, 0), 5)   # 검  # boxes[0] -> x1, y1 , 좌하단
# cv2.circle(binary_crop_image, boxes[1], 1, (255, 0, 0), 5) # 파  # boxes[1] -> x2, y1 , 좌상단
# cv2.circle(binary_crop_image, boxes[2], 1, (0, 255, 0), 5) # 녹  # boxes[2] -> x2, y2 , 우상단
# cv2.circle(binary_crop_image, boxes[3], 1, (0, 0, 255), 5) # 적  # boxes[3] -> x1, y2 , 우하단

# cv2.imshow("perspective_image2",binary_crop_image)

# W = rect[1][1]       # 90도 돌아간거로 인식되서 rect[1][1]이 width
# H = rect[1][0]

# pts1 = np.float32([ [boxes[1]], [boxes[0]], [boxes[2]], [boxes[3]] ])   # 좌상, 좌하, 우상, 우하
# pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

# M = cv2.getPerspectiveTransform(pts1, pts2)

# # crop 한 이미지 회전시키는 함수
# def getSubImage(rect, src):
#     center, size, theta = rect   # Get center, size, and angle from rect 
#     center, size = tuple(map(int, center)), tuple(map(int, size))     # Convert to int
#     M = cv2.getRotationMatrix2D( center, theta, 1)    # 사각형 회전하기 위한 정보 추출, center/반시계방향 회전각도/추가적인 확대비율
#     dst = cv2.warpAffine(src, M, (0,0) )       # Affine 이동변환(회전) , img/2x3어파인변환행렬/결과영상크기
#     out = cv2.getRectSubPix(dst, size, center)        # frame/(w,h)/(c_x,c_y)
#     return out

# if rect[2] < 10:
#     # Extract subregion
#     out = getSubImage(rect, binary_crop_image)
#     cv2.imshow('crop_image2', out)

# else:
#     crop_image = cv2.warpPerspective(binary_crop_image, M, (int(W),int(H)))
#     cv2.imshow("crop_image2", crop_image)


cv2.waitKey(0)
cv2.destroyAllWindows()







