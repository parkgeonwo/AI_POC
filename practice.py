import numpy as np
import cv2
import math

############## 이미지 전처리 (이미지 업로드, 이진화, blur, canny )

upload_image = cv2.imread("/home/matrix/Desktop/code/AI_POC/document/car_document3.jpg", cv2.IMREAD_GRAYSCALE)
upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
frame_height,frame_width = upload_image.shape

ret, binary_image = cv2.threshold(upload_image, 240, 255, cv2.THRESH_BINARY)   # 이진화

gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
# image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
# image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
canny_image = cv2.Canny(gblur_image, 75,200, True)



cv2.imshow("upload_image", upload_image)
cv2.imshow("gblur_image", gblur_image)
cv2.imshow("binary_image", binary_image)
cv2.imshow("canny_image", canny_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

