from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import urllib.request
import os
from werkzeug.utils import secure_filename
from Car_document_ocr import OCR

app = Flask(__name__)      # 플라스크 불러오기
app.debug=True             # debug mode on

# UPLOAD_FOLDER = 'static/uploads/'
UPLOAD_FOLDER = './flask/flaskapp/static/uploads/'    # upload image 경로 지정
app.secret_key = "secret key"                       # secret_key 설정
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER         # app dict의 sub class인 config에서 UPLOAD_FOLDER 내용 수정

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])        # upload 확장자 종류를 모은 set

def allowed_file(filename):     # ALLOWED_EXTENSIONS에 들어있는 확장자와 '.' 을 포함하는지 여부를 True, False로 return
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	# rsplit('.',1) 은 오른쪽에서부터 .을 기준으로 2개로 나누고 왼쪽부터 리스트 반환

@app.route('/')         # Route가 '/' 경로로 
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
	if 'upload_image' not in request.files:        # request.files에 upload_image이 없다면 
		flash('No file part')              # flash에 meassage를 담기
		# print(request.url)
		return redirect(request.url)       # request.url

	file = request.files['upload_image']       # html에서 upload_image 받아오기

	if file.filename == '':           # filename이 ''라면
		flash('No image selected for uploading')
		return redirect(request.url)

	if file and allowed_file(file.filename):     # file이 있고 & file이름에 확장자와 .이 있으면
		filename = secure_filename(file.filename)  # 파일명을 암호화 한 후 임시 위치에 저장
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))    # os.path.join은 합쳐주는 method / file을 경로에 save
		# print(app.config['UPLOAD_FOLDER'], filename)
		# print('upload_image filename: ' + filename)
		flash('Upload Image')

		upload_image_path = "./flask/flaskapp/static/uploads/"
		car_ocr = OCR(upload_image_path=upload_image_path+filename)
		car_ocr.img_process()
		ocr_result = car_ocr.ocr_process()
		ocr_columns = car_ocr.ocr_columns    # columns의 이름 전달
		columns_length_list = []             # columns의 번호 전달

		for i in range( len(ocr_columns) ):
			columns_length_list.append(i)

		return render_template('index.html', filename=filename, easyocr_list=ocr_result, ocr_columns=ocr_columns,
								columns_length_list=columns_length_list )     # filename을 index.html에 전달

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	# print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host = "0.0.0.0")   # 127.0.0.1 == localhost

