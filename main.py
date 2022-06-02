
from Car_document_ocr import OCR


car_ocr = OCR( upload_image_path="/home/matrix-5/Desktop/code/AI_POC/document/car_document.jpg" )

car_ocr.crop_image_save(save_path = "./crop_image/")

print( car_ocr.ocr_data_save(image_path = "./crop_image/", ocr_type="easyocr") )






