import numpy as np
import cv2
import math

############## 이미지 전처리 (이미지 업로드, 이진화, blur, canny )

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document3.jpg", cv2.IMREAD_GRAYSCALE)
# upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
frame_height,frame_width = upload_image.shape

ret, binary_image = cv2.threshold(upload_image, 240, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
# image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
canny_image = cv2.Canny(gblur_image, 75,200, True)



############### 큰박스 3개 검출

cnts, hierarchy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적
cnt = cnts[:3]  # 가장 큰 BOX 3개만 검출




############## 큰 박스 3개 좌상단 순서대로 정렬

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





###### n번 box 추출

n = 0

rect = rect_list3[n]
r = cv2.boxPoints(rect)
copy_r = r.copy()

if rect[2] > 10:        # 각도가 10도이상이라면 
    r[0], r[1], r[2], r[3] = copy_r[3],copy_r[0],copy_r[1],copy_r[2] # 좌표값 순서를 통일하기 위해 변경


box = np.int0(r)

upload_image = cv2.cvtColor(upload_image, cv2.COLOR_GRAY2BGR)
upload_image = cv2.drawContours(upload_image, [box], -1, (255,0,0), 3)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

cv2.imshow("upload_image",upload_image)


boxes = [tuple(i) for i in box]

if rect[2] > 10:
    W = rect[1][1]
    H = rect[1][0]
else:
    W = rect[1][0]
    H = rect[1][1]


pts1 = np.float32([ [boxes[1]], [boxes[0]], [boxes[2]], [boxes[3]] ])   # 좌상, 좌하, 우상, 우하
pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

M = cv2.getPerspectiveTransform(pts1, pts2)

crop_image = cv2.warpPerspective(upload_image, M, (int(W),int(H)))     # 원근변환
cv2.imshow("crop_image",crop_image)


crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
ret, binary_image2 = cv2.threshold(crop_image, 240, 255, cv2.THRESH_BINARY)   # 이진화






##################### 큰박스안에서 작은 박스찾기

cnts, hierarchy = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적
# print(len(cnts))

##################### 작은박스들 좌상단 순서로 정렬

rect_list = []    # rect를 담기위한 list

for i in range(len(cnts)):
    rect_list.append(cv2.minAreaRect(cnts[i]))   # rect_list에 검출된 rect 수만큼 추가

rect_list2 = []
rect_list3 = []


if n == 0 :
    height_list = [0, H*0.1617, H*0.2318, H*0.3073, H*0.3733, H*0.442, H*0.5054, H*0.5701, H ]
    # 120, 172, 228, 277, 328, 375, 423, 734    / 742, 1173 = height, width

elif n == 1:
    height_list = [0,H*0.05853, H*0.1146, H*0.1573, H*0.1988,H*0.239,H*0.283,H*0.3598,H ]
    # 0, 48, 94, 129, 163, 196, 232, 295  / 820, 569 = height, width


for j in range(len(height_list)):
    rect_list2 = [rect_list[i] for i in range(len(rect_list)) if rect_list[i][0][1] >= height_list[j] and rect_list[i][0][1] <= height_list[j+1] ]
    rect_list2.sort(key=lambda tup: tup[0][0])
    for i in rect_list2:
        rect_list3.append(i)



#################### k번 박스 추출

image_number = 1

for k in range(len(rect_list3)):

    rect = rect_list3[k]
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
        image_number += 1


    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.waitKey(0)
# cv2.destroyAllWindows()

