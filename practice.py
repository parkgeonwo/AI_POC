import numpy as np
import cv2

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document3.jpg", cv2.IMREAD_GRAYSCALE)
# src = cv2.resize( src, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )

ret, binary_image = cv2.threshold(upload_image, 230, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
# image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
canny_image = cv2.Canny(gblur_image, 75,200, True)


######## 큰박스 

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




### 큰박스안에서 작은 박스찾기

cnts, hierarchy = cv2.findContours(perspective_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

# print(len(cnts))          # 26개의 box가 검출되어야한다.

rect_list = []    # rect를 담기위한 list

for i in range(len(cnts)):      # cnts 수만큼 
    rect_list.append(cv2.minAreaRect(cnts[i]))   # rect_list에 검출된 rect 수만큼 추가

# sorted_rect_list = sorted(rect_list, key=lambda tup: tup[0][1])      # c_y의 크기에 따라 정렬    # 위에서부터 box를 가져오기위해

rect_list2 = []
rect_list3 = []

height_list = [0, H*0.1617, H*0.2318 ]


rect_list2 = [rect_list[i] for i in range(len(rect_list)) if rect_list[i][0][1] <= H*0.1617 ]
rect_list2.sort(key=lambda tup: tup[0][0])
print(rect_list2)

rect_list2 = [rect_list[i] for i in range(len(rect_list)) if rect_list[i][0][1] >= H*0.1617 and rect_list[i][0][1] <= H*0.2318 ]     # c_y가 특정 범위 내에 있는 rect_list를 찾아주고 rect_list2에 담아줌
rect_list2.sort(key=lambda tup: tup[0][0])
print(rect_list2)




# if rect_list[i][0][1] >= H*0.1617 and rect_list[i][0][1] <= H*0.2318

# for i in range(len(cnts)):

#     # print( [i for i in rect_list if rect_list[i][0][1] >= H*0.1617 and rect_list[i][0][1] <= H*0.2318] )
#     # print( rect_list[rect_list[i][0][1] <= H*0.1617] )

#     if rect_list[i][0][1] <= H*0.1617:     # 특정 높이보다 낮은 rect_list를 찾아주고
#         rect_list2.append( rect_list[i] )    # 그걸 list2에 더해주고
#         rect_list2.sort(key=lambda tup: tup[0][0])     # c_x의 크기에 따라 정렬 (오름차순)
#         for i in range(len(rect_list2)):
#             rect_list3.append(rect_list2[i])     # 정렬된 list2의 값을 list3에 더해줌
#         rect_list2 = []       # rect_list2 초기화
    
#     elif rect_list[i][0][1] <= H*0.2318:
#         rect_list2.append( rect_list[i] )    # 그걸 list2에 더해주고
#         rect_list2.sort(key=lambda tup: tup[0][0])     # c_x의 크기에 따라 정렬 (오름차순)
#         for i in range(len(rect_list2)):
#             rect_list3.append(rect_list2[i])     # 정렬된 list2의 값을 list3에 더해줌
#         rect_list2 = []


# print(rect_list3)
    












rect = cv2.minAreaRect(cnts[0])  # largest 중 하나를 직사각형 형태로 return = (c_x,c_y) / (width, height) / angle of rotation


r = cv2.boxPoints(rect)
copy_r = r.copy()

if rect[2] > 10:        # 각도가 10도이상이라면, 좌표값 순서를 통일하기 위해 변경
    r[0], r[1], r[2], r[3] = copy_r[3],copy_r[0],copy_r[1],copy_r[2] 

box = np.int0(r)     # r을 int로 변환

perspective_image = cv2.cvtColor(perspective_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(perspective_image, [box], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

# # 4개의 점 다른색으로 표시
boxes = [tuple(i) for i in box]
cv2.circle(perspective_image, boxes[0], 1, (0, 0, 0), 5)   # 검  # boxes[0] -> x1, y1 , 좌하단
cv2.circle(perspective_image, boxes[1], 1, (255, 0, 0), 5) # 파  # boxes[1] -> x2, y1 , 좌상단
cv2.circle(perspective_image, boxes[2], 1, (0, 255, 0), 5) # 녹  # boxes[2] -> x2, y2 , 우상단
cv2.circle(perspective_image, boxes[3], 1, (0, 0, 255), 5) # 적  # boxes[3] -> x1, y2 , 우하단

cv2.imshow("perspective_image",perspective_image)

W = rect[1][1]       # 90도 돌아간거로 인식되서 rect[1][1]이 width
H = rect[1][0]

pts1 = np.float32([ [boxes[1]], [boxes[0]], [boxes[2]], [boxes[3]] ])   # 좌상, 좌하, 우상, 우하
pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

M = cv2.getPerspectiveTransform(pts1, pts2)

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)    # center/반시계방향 회전각도/추가적인 확대비율
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, (0,0) )       # Affine 이동변환 , img/2x3어파인변환행렬/결과영상크기
    # src.shape[1],src.shape[0]
    out = cv2.getRectSubPix(dst, size, center)        # frame/(w,h)/(c_x,c_y)
    return out


if rect[2] < 10:
    # Extract subregion
    out = getSubImage(rect, perspective_image)
    # Save image
    cv2.imshow('out.jpg', out)

else:
    crop_image = cv2.warpPerspective(perspective_image, M, (int(W),int(H)))
    cv2.imshow("crop_image", crop_image)


cv2.waitKey(0)
cv2.destroyAllWindows()







