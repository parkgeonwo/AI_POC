import cv2
import numpy as np
import math
import easyocr
import pytesseract
import pandas as pd


class OCR():
    def __init__(self,upload_image_path):
        self.img = cv2.imread(upload_image_path)

        # 가장 큰 box 몇개 추출할지 지정
        self.detect_large_box_num = 3

        # box마다 중심점 높이의 범위를 지정해주는 list
        self.box_center_list = [[0, 0.5, 1],    # 가장 큰 box 3개
            [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],    # 가장 큰 박스 3개중 1번째 박스의 높이 리스트
            [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],   # 가장 큰 박스 3개중 2번째 박스의 높이 리스트
            None ]  # 가장 큰 박스 3개중 3번째 박스의 높이 리스트 

        self.crop_img_save_path = "./crop_image/"


    def img_process(self):
        img_height,img_width, c = self.img.shape
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gblur_img = cv2.GaussianBlur(gray_img,(5,5),0)
        canny_img = cv2.Canny(gblur_img, 75,200, True)
        ret, binary = cv2.threshold(canny_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        large_box_cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(large_box_cnts) < self.detect_large_box_num:      # box 검출 수가 적으면
            errer_message = "len(large_box_cnts) is too small"
            print(errer_message)

        large_box_cnts = sorted(large_box_cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적 # 면적크기에 따라 정렬

        # for i in range(self.detect_large_box_num):    # large_box_cnts 크기별로 정렬한 box 3위까지 그려주기
        #     cv2.drawContours(self.img, [large_box_cnts[i]], 0, (0, 0, 255), 2)
        #     cv2.putText(self.img, str(i), tuple(large_box_cnts[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

        large_box_list = []    # 큰 box 3개 좌표 담을 list

        for i in range(self.detect_large_box_num):
            reshape_size_x, reshape_size_y = large_box_cnts[i].shape[0],large_box_cnts[i].shape[2]   # reshape할 size 정의
            points = large_box_cnts[i].reshape(reshape_size_x, reshape_size_y)             # reshape
            large_input_points = np.zeros((4,2), dtype = "float32")                    # 틀 만들어 놓기

            points_sum = points.sum(axis = 1)
            large_input_points[0] = points[np.argmin(points_sum)]   # top left
            large_input_points[3] = points[np.argmax(points_sum)]   # bottom right

            points_diff = np.diff(points, axis=1)
            large_input_points[1] = points[np.argmin(points_diff)]   # top right
            large_input_points[2] = points[np.argmax(points_diff)]   # bottom left

            large_box_list.append(large_input_points)

        large_box_list2 = []
        large_box_list3 = []

        # 큰 box들의 중심 좌표를 구분할 높이 리스트
        large_box_height_list = [ i*img_height for i in self.box_center_list[0] ]

        # 같은 높이 범위에 있는 box들을 large_box_list2에 담고 좌측->우측으로 정렬한다음 
        # 들어있는 box들을 large_box_list3에 append
        for j in range(len(large_box_height_list)-1):
            large_box_list2 = [large_box_list[i] for i in range(len(large_box_list))
                    if ((large_box_list[i][0][1]+large_box_list[i][3][1])/2) >= large_box_height_list[j] and ((large_box_list[i][0][1]+large_box_list[i][3][1])/2) <= large_box_height_list[j+1] ]
            large_box_list2.sort(key=lambda x: x[0][0])
            for i in large_box_list2:
                large_box_list3.append(i)


        ############ 추출된 큰 box를 하나씩 원근변환하고 큰box안에서 작은box들을 crop하여 저장

        for k in range(self.detect_large_box_num):
            
            ############## k번째 큰box 원근변환
            large_input_points2 = large_box_list3[k]

            (top_left, top_right, bottom_left, bottom_right) = large_input_points2
            bottom_width = np.sqrt( ((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2) )
            top_width = np.sqrt( ((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2) )
            right_height = np.sqrt( ((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2) )
            left_height = np.sqrt( ((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2) )

            crop_max_width = max( int(bottom_width), int(top_width) )
            crop_max_height = max( int(right_height), int(left_height) )

            converted_points = np.float32( [[0,0], [crop_max_width,0], [0,crop_max_height], [crop_max_width, crop_max_height]] )

            matrix = cv2.getPerspectiveTransform(large_input_points2, converted_points)
            crop_img = cv2.warpPerspective( self.img , matrix, (crop_max_width,crop_max_height))

            ################## k번째 큰박스(crop_image)안에서 작은 박스찾기
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            ret, small_binary_image = cv2.threshold(crop_img, 240, 255, cv2.THRESH_BINARY)   # 이진화

            small_box_cnts, hierarchy = cv2.findContours(small_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # image / mode / method
            small_box_cnts = sorted(small_box_cnts, key=cv2.contourArea, reverse=True) # contourArea : contour가 그린 면적
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

            ##################### 작은박스들 좌상단->우하단 순서로 정렬

            small_rect_list = []    # rect를 담기위한 list

            for i in range(len(small_box_cnts)):
                small_rect_list.append(cv2.minAreaRect(small_box_cnts[i]))   # small_rect_list에 검출된 rect 수만큼 추가

            small_rect_list2 = []
            small_rect_list3 = []

            for i in range(self.detect_large_box_num):
                if self.box_center_list[k+1] == None:
                    small_box_height_list = [0,0.5,1]
                elif k == i:
                    small_box_height_list = [ j*crop_max_height for j in self.box_center_list[k+1] ]


            for j in range(len(small_box_height_list)-1):
                small_rect_list2 = [small_rect_list[i] for i in range(len(small_rect_list)) if small_rect_list[i][0][1] >= small_box_height_list[j] and small_rect_list[i][0][1] <= small_box_height_list[j+1] ]
                small_rect_list2.sort(key=lambda tup: tup[0][0])
                for i in small_rect_list2:
                    small_rect_list3.append(i)

            #################### m번 박스 추출

            image_num = 0
            for m in range(len(small_rect_list3)):

                rect = small_rect_list3[m]
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

                if W > (crop_max_width*3/100) and H >(crop_max_height*3/100) and W>H:
                    
                    cv2.drawContours(crop_img, [box2], -1, (255,0,0), 1)      # img / 좌표 / 외곽선 index, -1하면 모든 외곽선 그리기 / 색 / 굵기
                    ocr_crop_image = crop_img[ box2[1][1] : box2[1][1] + math.ceil(H) , box2[1][0] : box2[1][0] + math.ceil(W) ]
                    cv2.imwrite(self.crop_img_save_path+"ocr_crop_image_{}_{}.png".format(k,image_num), ocr_crop_image)

                    # cv2.imshow("crop_img", crop_img)
                    # cv2.imshow("OCR_crop_image_{}_{}".format(k,image_num), ocr_crop_image)

                    image_num += 1

                    # # 하나하나 띄우기
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()


    # ocr 결과 data를 저장
    def ocr_process(self,ocr_type = 2):
        # image가 들어있는 path / ocr 적용 엔진 지정

        # crop box num list 지정
        self.box1_ocr_num_list = [0,2,4,6,8,10,12,14,16,18,20,22 ]
        self.box2_ocr_num_list = [1,3,5,7,9,11,13,15,17,19,21]
        ocr_list = []

        ### tesseeract ocr
        if ocr_type == 1:
            for i in range(2):
                if i == 0:
                    box_num_list = self.box1_ocr_num_list
                if i == 1:
                    box_num_list = self.box2_ocr_num_list

                for k in box_num_list:
                    path = self.crop_img_save_path + "ocr_crop_image_"+str(i)+"_"+str(k)+".png"
                    crop_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # rgb_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

                    ret, binary_image = cv2.threshold(crop_image, 210, 255, cv2.THRESH_BINARY)   # 이진화

                    text = pytesseract.image_to_string(binary_image, lang="kor+eng")

                    result = text.strip()
                    result = result.replace(" ","")
                    result = result.replace("\n"," ")

                    ocr_list.append(result)

            ocr_result_list = ocr_list

        ##### easyocr ocr
        if ocr_type == 2:
            reader = easyocr.Reader(['ko', 'en'])

            for i in range(2):
                if i == 0:
                    box_num_list = self.box1_ocr_num_list
                if i == 1:
                    box_num_list = self.box2_ocr_num_list

                for k in box_num_list:
                    path = self.crop_img_save_path + "ocr_crop_image_"+str(i)+"_"+str(k)+".png"
                    # result = reader.readtext(path)
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    ret, binary_image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)   # 이진화
                    # blur
                    gblur_image = cv2.GaussianBlur(binary_image, (3,3), 0)      # 전체적으로 밀도가 동일한 노이즈, 백색 노이즈를 제거하는 기능

                    # cv2.imshow("OCR_crop_image", gblur_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    result = reader.readtext(gblur_image)

                    ocr_word_list = []

                    for j in result:
                        ocr_word = j[1].strip()
                        ocr_word = ocr_word.replace(" ","")
                        ocr_word_list.append(ocr_word)
                    
                    ocr_list.append(ocr_word_list)

            # easyocr을 ocr각 값들을 list로 반환하기 때문에 하나의 string으로 바꿔서 ocr_list2에 저장
            ocr_list2 = []

            def sum_string(sum_list):
                sum_str=''
                for i in range(len(sum_list)):
                    sum_str += sum_list[i] + ' '
                return sum_str

            for i in ocr_list:
                sum_str = sum_string(i)
                ocr_list2.append(sum_str)

            ocr_result_list = ocr_list2
        
        # ocr결과 데이터를 csv파일로 저장
        self.ocr_columns = [ 'info' , '자동차등록번호' , '차종' ,'용도' ,'차명' ,'형식 및 모델연도' ,'차대번호' ,'원동기형식' ,
                '사용본거지' ,'성명(명칭)' ,'주민(법인)등록번호' ,'주소' ,
                '제원관리번호(형식승원번호)' ,'길이' ,'너비' ,'높이' ,'총중량' ,'배기량' ,'정격출력' ,'승차정원' ,
                '최대적재량' ,'기통수' ,'연료의 종류'  ]
        ocr_data_frame = pd.DataFrame(columns=self.ocr_columns)
        ocr_data_frame.loc[0]= ocr_result_list

        ocr_data_frame.to_csv("./result.csv")

        return ocr_result_list

