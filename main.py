
from imageio import save
from Car_document_ocr import OCR


car_ocr = OCR( upload_image_path="./document/car_document.jpg" )

car_ocr.crop_image_save(save_path = "./crop_image/",show_image=False)

data =  car_ocr.ocr_data_save(image_path = "./crop_image/", ocr_type="easyocr") 
print(data)

df = car_ocr.save_csv(data,save_csv=True)
print(df)

# import pandas as pd

# ocr_data_dict = { 'info' : [], '자동차등록번호' : [], '차종' : [],'용도' : [],'차명' : [],'형식 및 모델연도' : [],'차대번호' : [],'원동기형식' : [],
#                 '사용본거지' : [],'성명(명칭)' : [],'주민(법인)등록번호' : [],'주소' : [],
#                 '제원관리번호(형식승원번호)' : [],'길이' : [],'너비' : [],'높이' : [],'총중량' : [],'배기량' : [],'정격출력' : [],'승차정원' : [],
#                 '최대적재량' : [],'기통수' : [],'연료의 종류' : [] } 


# ocr_data_frame = pd.DataFrame(ocr_data_dict) 

# print(ocr_data_frame) 


# ball = pd.DataFrame(ball2, index = [0])      # 오류뜨면 



# ocr_columns = [ 'info' , '자동차등록번호' , '차종' ,'용도' ,'차명' ,'형식 및 모델연도' ,'차대번호' ,'원동기형식' ,
#                 '사용본거지' ,'성명(명칭)' ,'주민(법인)등록번호' ,'주소' ,
#                 '제원관리번호(형식승원번호)' ,'길이' ,'너비' ,'높이' ,'총중량' ,'배기량' ,'정격출력' ,'승차정원' ,
#                 '최대적재량' ,'기통수' ,'연료의 종류'  ]

# import pandas as pd
# #빈 DataFrame 생성
# df1 = pd.DataFrame(columns=ocr_columns)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# # pd.set_option('display.width', None)
# # pd.set_option('display.max_colwidth', -1)


# df1.loc[0]= data
# print(df1)







