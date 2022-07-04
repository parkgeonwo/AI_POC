import cv2
import numpy as np



class Image_processing():
    def __init__(self, path):
        self.upload_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # src = cv2.resize( src, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )

        ret, self.binary_image = cv2.threshold(self.upload_image, 230, 255, cv2.THRESH_BINARY)   # 이진화

        self.gblur_image = cv2.GaussianBlur(self.binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
        # image_binary = cv2.bilateralFilter(image_binary, 9,75,75)
        # image_binary = cv2.edgePreservingFilter(image_binary, flags=1, sigma_s=45, sigma_r=0.2)
        self.canny_image = cv2.Canny(self.gblur_image, 75,200, True)

    def findcontours(self, image, sort_box_index):
        cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

        rect = cv2.minAreaRect(cnts[sort_box_index])  # largest 중 하나를 직사각형 형태로 return = (c_x,c_y) / (width, height) / angle of rotation
        r = cv2.boxPoints(rect)
        box = np.int0(r)

        self.upload_image = cv2.cvtColor(self.upload_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.upload_image, [box], -1, (255,0,0), 2)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기

        # # 4개의 점 다른색으로 표시
        boxes = [tuple(i) for i in box]
        cv2.circle(self.upload_image, boxes[0], 1, (0, 0, 0), 5)   # 검  # boxes[0] -> x1, y1 , 좌상단
        cv2.circle(self.upload_image, boxes[1], 1, (255, 0, 0), 5) # 파  # boxes[1] -> x2, y1 , 우상단
        cv2.circle(self.upload_image, boxes[2], 1, (0, 255, 0), 5) # 녹  # boxes[2] -> x2, y2 , 우하단
        cv2.circle(self.upload_image, boxes[3], 1, (0, 0, 255), 5) # 적  # boxes[3] -> x1, y2 , 좌하단

        W = rect[1][1]       # 90도 돌아간거로 인식되서 rect[1][1]이 width
        H = rect[1][0]

        pts1 = np.float32([ [boxes[0]], [boxes[3]], [boxes[1]], [boxes[2]] ])   # 좌상, 좌하, 우상, 우하
        pts2 = np.float32([ [0,0], [0,H], [W,0], [W,H] ])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        perspective_image = cv2.warpPerspective(self.binary_image, M, (int(W),int(H)))

        return int(W),int(H), perspective_image

    def extract_ocr_image(self, W, H, perspective_image, path):
        left_col_x1 = int(W*0.157)
        left_col_x2 = int(W*0.528)
        col_y1 = int(H*0.182)
        row_height = 45
        row_interval = 3

        right_col_x1 = int(W*0.664)
        right_col_x1_1 = int(W*0.791)
        right_col_x1_2 = int(W*0.87)
        right_col_x2 = int(W*0.998)

        cv2.imshow("perspective_image",perspective_image)

        for i in range(0,6):
            if i in [3,5]:
                x1, y1 = left_col_x1, col_y1 + row_height *i + row_interval *i
                x2, y2 = right_col_x2, col_y1 + row_height *(i+1) + row_interval *i
                crop_img = perspective_image[y1:y2, x1:x2]
                cv2.imshow("crop_img_%d" %(i), crop_img)
                # cv2.imshow("perspective_image",perspective_image)
                cv2.waitKey(0)
            else:
                x1, y1 = left_col_x1, col_y1 + row_height *i + row_interval *i
                x2, y2 = left_col_x2, col_y1 + row_height *(i+1) + row_interval *i
                crop_img = perspective_image[y1:y2, x1:x2]
                cv2.imshow("crop_img_%d" %(i), crop_img)
                cv2.waitKey(0)

        for i in range(0,6):
            if i == 0:
                x1, y1 = right_col_x1, col_y1 + row_height *i + row_interval *i
                x2, y2 = right_col_x1_1, col_y1 + row_height *(i+1) + row_interval *i
                crop_img = perspective_image[y1:y2, x1:x2]
                cv2.imshow("crop_img_%d" %(i+6), crop_img)
                cv2.waitKey(0)
            elif i == 1:
                x1, y1 = right_col_x1_2, col_y1 + row_height *(i-1) + row_interval *(i-1)
                x2, y2 = right_col_x2, col_y1 + row_height *(i) + row_interval *(i-1)
                crop_img = perspective_image[y1:y2, x1:x2]
                cv2.imshow("crop_img_%d" %(i+6), crop_img)
                cv2.waitKey(0)
            elif i in [2,3,5]:
                x1, y1 = right_col_x1, col_y1 + row_height *(i-1) + row_interval *(i-1)
                x2, y2 = right_col_x2, col_y1 + row_height *(i) + row_interval *(i-1)
                crop_img = perspective_image[y1:y2, x1:x2]
                if i == 5:
                    cv2.imshow("crop_img_%d" %(i+5), crop_img)
                    cv2.waitKey(0)
                else:
                    cv2.imshow("crop_img_%d" %(i+6), crop_img)
                    cv2.waitKey(0)


image = Image_processing(path = "./document/car_document3.jpg")

W, H, perspective_image = image.findcontours(image.canny_image, 0)

image.extract_ocr_image(W, H, perspective_image, path="./")

# cv2.imshow("image", image.upload_image)
# cv2.imshow("image2", image.binary_image)
# cv2.imshow("image3", image.canny_image)

cv2.waitKey(0)
cv2.destroyAllWindows()





