import pytesseract
import cv2 

path = "/home/matrix/Desktop/code/AI_POC/example/OCR_crop_image_1.png"
# image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# ret, binary_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)   # 이진화

image = cv2.imread(path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# use Tesseract to OCR the image 
text = pytesseract.image_to_string(rgb_image, lang='kor')
result = text.strip()
print(result,len(result))

# cv2.imshow("img",binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




