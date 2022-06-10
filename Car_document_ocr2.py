import easyocr
import pytesseract
import cv2
import numpy as np
import pandas as pd

class OCR():
    def __init__(self,upload_image_path):

        ############## 이미지 전처리 (이미지 업로드, 이진화, blur, canny )
        self.upload_image = cv2.imread(upload_image_path)

        self.upload_image_height,self.upload_image_width, c = self.upload_image.shape

        self.extend_size = 1.5

        if self.upload_image_height < 1000 or self.upload_image_width < 1000:
            self.upload_image = cv2.resize( self.upload_image, None, fx = self.extend_size, fy = self.extend_size, interpolation = cv2.INTER_LANCZOS4 )
        
        self.upload_image_gray = cv2.cvtColor(self.upload_image, cv2.COLOR_BGR2GRAY)

        # upload_image = cv2.resize( upload_image, None, fx = 0.7, fy = 0.7, interpolation = cv2.INTER_AREA )
        self.frame_height,self.frame_width = self.upload_image_gray.shape

        ret, self.binary_image = cv2.threshold(self.upload_image_gray, 240, 255, cv2.THRESH_BINARY)   # 이진화

        self.gblur_image = cv2.GaussianBlur(self.binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능
        self.canny_image = cv2.Canny(self.gblur_image, 75,200, True)

        self.detect_box_num = 3
        self.box_center_list = [[0, 0.5, 1],    # 가장 큰 box 3개
                    [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],    # 가장 큰 박스 3개중 1번째 박스의 높이 리스트
                    [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],   # 가장 큰 박스 3개중 2번째 박스의 높이 리스트
                    None ]  # 가장 큰 박스 3개중 3번째 박스의 높이 리스트 

    def biggest_contour(self, contours):
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

    def crop_image_save(self, save_path, show_image = False ):
        
        self.save_path = save_path
        self.processed_image = self.canny_image

        ############### 큰박스 3개 검출
        cnts, hierarchy = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

        ######## rect_list에 cnts를 좌상단->우하단 순서로 만들기위해서 좌표를 담아주고 정렬

        rect_list = []
        self.upload_image = cv2.cvtColor(self.upload_image_gray, cv2.COLOR_GRAY2BGR)

        for i in range(self.detect_box_num):
            contours = cnts[i:]
            biggest = self.biggest_contour(contours)

            cv2.drawContours(self.upload_image, [biggest], -1, (255,0,0),3 )

            points = biggest.reshape(4,2)
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

        frame_height, frame_width = self.processed_image.shape

        height_list = [ i*frame_height for i in self.box_center_list[0] ]

        for j in range(len(height_list)-1):
            rect_list2 = [rect_list[i] for i in range(len(rect_list)) if ((rect_list[i][0][1]+rect_list[i][3][1])/2) >= height_list[j] and ((rect_list[i][0][1]+rect_list[i][3][1])/2) <= height_list[j+1] ]
            rect_list2.sort(key=lambda x: x[0][0])
            for i in rect_list2:
                rect_list3.append(i)


        ################## 각 좌표에따라 원근변환

        for k in range(self.detect_box_num):
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
            crop_image = cv2.warpPerspective( self.upload_image , matrix, (max_width,max_height))

            if show_image == True:
                cv2.imshow("upload_image", self.upload_image)
                cv2.imshow("crop_image", crop_image)



            ##################### 큰박스안에서 작은 박스찾기
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            ret, small_binary_image = cv2.threshold(crop_image, 240, 255, cv2.THRESH_BINARY)   # 이진화

            small_cnts, hierarchy = cv2.findContours(small_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
            small_cnts = sorted(small_cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적

            ##################### 작은박스들 좌상단 순서로 정렬

            small_rect_list = []    # rect를 담기위한 list

            for i in range(len(small_cnts)):
                small_rect_list.append(cv2.minAreaRect(small_cnts[i]))   # small_rect_list에 검출된 rect 수만큼 추가

            small_rect_list2 = []
            small_rect_list3 = []

            for i in range(self.detect_box_num):
                if self.box_center_list[k+1] == None:
                    sub_height_list = [0,0.5,1]
                elif k == i:
                    sub_height_list = [ j*max_height for j in self.box_center_list[k+1] ]

            # print(sub_height_list)


            for j in range(len(sub_height_list)-1):
                small_rect_list2 = [small_rect_list[i] for i in range(len(small_rect_list)) if small_rect_list[i][0][1] >= sub_height_list[j] and small_rect_list[i][0][1] <= sub_height_list[j+1] ]
                small_rect_list2.sort(key=lambda tup: tup[0][0])
                for i in small_rect_list2:
                    small_rect_list3.append(i)

            # print(len(small_rect_list3))

            #################### m번 박스 추출

            image_num = 0
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
                    
                    cv2.drawContours(crop_image2, [box2], -1, (255,0,0), 1)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기
                    cv2.imshow("crop_image2", crop_image2)
                    OCR_crop_image = crop_image2[ box2[1][1]-3 : box2[1][1] + int(H)+3 , box2[1][0]-3 : box2[1][0] + int(W)+3 ]
                    
                    # cv2.imwrite(self.save_path+"OCR_crop_image_{}_{}.png".format(k,image_num), OCR_crop_image)

                    if show_image == True:
                        cv2.imshow("crop_image2", crop_image2)
                        cv2.imshow("OCR_crop_image_{}_{}".format(k,image_num), OCR_crop_image)

                    image_num += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ocr_data_save(self,image_path,ocr_type = "easyocr"):

        self.box1_ocr_num_list = [0,2,4,6,8,10,12,14,16,18,20,23 ]
        self.box2_ocr_num_list = [1,3,5,7,9,11,13,15,17,19,21]
        self.image_path = image_path
        ocr_list = []

        ### tesseeract ocr
        if ocr_type == "tesseract":
            for i in range(2):
                if i == 0:
                    box_num_list = self.box1_ocr_num_list
                if i == 1:
                    box_num_list = self.box2_ocr_num_list

                for k in box_num_list:
                    path = self.image_path + "OCR_crop_image_"+str(i)+"_"+str(k)+".png"
                    crop_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

                    ret, binary_image = cv2.threshold(crop_image, 210, 255, cv2.THRESH_BINARY)   # 이진화

                    text = pytesseract.image_to_string(binary_image, lang="kor+eng")

                    result = text.strip()
                    result = result.replace(" ","")
                    result = result.replace("\n"," ")

                    ocr_list.append(result)

        ##### easyocr ocr
        if ocr_type =="easyocr":
            reader = easyocr.Reader(['ko', 'en'])

            for i in range(2):
                if i == 0:
                    box_num_list = self.box1_ocr_num_list
                if i == 1:
                    box_num_list = self.box2_ocr_num_list

                for k in box_num_list:
                    path = self.image_path + "OCR_crop_image_"+str(i)+"_"+str(k)+".png"
                    result = reader.readtext(path)
                    # image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # ret, binary_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)   # 이진화
                    # result = reader.readtext(binary_image)

                    ocr_word_list = []

                    for j in result:
                        ocr_word = j[1].strip()
                        ocr_word = ocr_word.replace(" ","")
                        ocr_word_list.append(ocr_word)
                    
                    ocr_list.append(ocr_word_list)

        return ocr_list

    def save_csv(self,data, save_csv = False):
        self.ocr_columns = [ 'info' , '자동차등록번호' , '차종' ,'용도' ,'차명' ,'형식 및 모델연도' ,'차대번호' ,'원동기형식' ,
                '사용본거지' ,'성명(명칭)' ,'주민(법인)등록번호' ,'주소' ,
                '제원관리번호(형식승원번호)' ,'길이' ,'너비' ,'높이' ,'총중량' ,'배기량' ,'정격출력' ,'승차정원' ,
                '최대적재량' ,'기통수' ,'연료의 종류'  ]
        self.ocr_data_frame = pd.DataFrame(columns=self.ocr_columns)

        self.ocr_data_frame.loc[0]= data

        if save_csv == True:
            self.ocr_data_frame.to_csv("./result.csv")

        return self.ocr_data_frame





