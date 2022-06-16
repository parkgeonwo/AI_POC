# #app.py
# from flask import Flask, json, request, jsonify
# import os
# import urllib.request
# from werkzeug.utils import secure_filename
 
# app = Flask(__name__)
 
# app.secret_key = "secret_key"
 
# # UPLOAD_FOLDER = './static/uploads'
# UPLOAD_FOLDER = './flask_API/static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
# @app.route('/')
# def main():
#     return 'Homepage'
 
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # check if the post request has the file part
#     if 'upload_image' not in request.files:
#         resp = jsonify({'message' : 'No file part in the request'})
#         resp.status_code = 400
#         return resp

#     file = request.files['upload_image']

# 	if file.filename == '':           # filename이 ''라면
# 		resp = jsonify({'message' : 'No image selected for uploading'})
#         resp.status_code = 400
#         return resp

#     errors = {}
#     success = False

# 	if file and allowed_file(file.filename):
# 		filename = secure_filename(file.filename)
# 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# 		success = True
# 	else:
# 		errors[file.filename] = 'File type is not allowed'
 
#     if success and errors:
#         errors['message'] = 'File(s) successfully uploaded'
#         resp = jsonify(errors)
#         resp.status_code = 500
#         return resp
#     if success:
#         resp = jsonify({'message' : 'Files successfully uploaded'})
#         resp.status_code = 201
#         return resp
#     else:
#         resp = jsonify(errors)
#         resp.status_code = 500
#         return resp
 
# if __name__ == '__main__':
#     app.run(debug=True)






#app.py
from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from Car_document_ocr import OCR
 
app = Flask(__name__)
 
app.secret_key = "secret_key"
 
# UPLOAD_FOLDER = './static/uploads'
UPLOAD_FOLDER = './flask_API/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if "upload_image" not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    # files = request.files.getlist('files[]')
    file = request.files["upload_image"]
    
    if file.filename == '':
        resp = jsonify({'message': 'No image selected for uploading'})
        resp.status_code = 400
        return resp

    errors = {}
    ocr_result_dict = {}
    success = False

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        ### ocr
        upload_image_path = "./flask/flaskapp/static/uploads/"
        car_ocr = OCR( upload_image_path=upload_image_path + filename )
        car_ocr.crop_image_save(save_path = upload_image_path + "crop_image/" ,show_image=False)
        ocr_data =  car_ocr.ocr_data_save(image_path = upload_image_path + "crop_image/", ocr_type="easyocr")
        df = car_ocr.save_csv(ocr_data,save_csv=True)

        ocr_columns = car_ocr.ocr_columns    # columns의 이름 전달

        columns_length_list = []             # columns의 번호 전달
        for i in range( len(ocr_columns) ):
            columns_length_list.append(i)
        
        easyocr_list = car_ocr.easyocr_list  # easyocr 결과 전달

        for i in range(len(ocr_columns)):
            ocr_result_dict[ocr_columns[i]] = easyocr_list[i]
        
        success = True
    else:
        errors[file.filename] = 'File type is not allowed'
 
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        # resp = jsonify({'message' : 'Files successfully uploaded'})
        # resp = json.dumps({'message' : 'No file part in the request'})
        resp = jsonify({'message' : 'Files successfully uploaded', 'ocr_result_dict' : ocr_result_dict})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
if __name__ == '__main__':
    app.run(debug=True)

















# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
# import urllib.request
# import os
# from werkzeug.utils import secure_filename
# from Car_document_ocr import OCR

# app = Flask(__name__)      # 플라스크 불러오기
# app.debug=True             # debug mode on

# # UPLOAD_FOLDER = 'static/uploads/'
# UPLOAD_FOLDER = './flask/flaskapp/static/uploads/'    # upload image 경로 지정
# app.secret_key = "secret key"                       # secret_key 설정
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER         # app dict의 sub class인 config에서 UPLOAD_FOLDER 내용 수정

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])        # upload 확장자 종류를 모은 set

# def allowed_file(filename):     # ALLOWED_EXTENSIONS에 들어있는 확장자와 '.' 을 포함하는지 여부를 True, False로 return
# 	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# 	# rsplit('.',1) 은 오른쪽에서부터 .을 기준으로 2개로 나누고 왼쪽부터 리스트 반환

# @app.route('/')         # Route가 '/' 경로로 
# def index():
#     return render_template("index.html")

# @app.route('/', methods=['POST'])
# def upload_image():
# 	if 'upload_image' not in request.files:        # request.files에 upload_image이 없다면 
# 		flash('No file part')              # flash에 meassage를 담기
# 		# print(request.url)
# 		return redirect(request.url)       # request.url

# 	file = request.files['upload_image']       # html에서 upload_image 받아오기

# 	if file.filename == '':           # filename이 ''라면
# 		flash('No image selected for uploading')
# 		return redirect(request.url)

# 	if file and allowed_file(file.filename):     # file이 있고 & file이름에 확장자와 .이 있으면
# 		filename = secure_filename(file.filename)  # 파일명을 암호화 한 후 임시 위치에 저장
# 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))    # os.path.join은 합쳐주는 method / file을 경로에 save
# 		# print(app.config['UPLOAD_FOLDER'], filename)
# 		# print('upload_image filename: ' + filename)
# 		flash('Upload Image')

# 		### ocr
# 		upload_image_path = "./flask/flaskapp/static/uploads/"
# 		car_ocr = OCR( upload_image_path=upload_image_path + filename )
# 		car_ocr.crop_image_save(save_path = upload_image_path + "crop_image/" ,show_image=False)
# 		ocr_data =  car_ocr.ocr_data_save(image_path = upload_image_path + "crop_image/", ocr_type="easyocr")
# 		df = car_ocr.save_csv(ocr_data,save_csv=True)
# 		# print(data)

# 		ocr_columns = car_ocr.ocr_columns    # columns의 이름 전달

# 		columns_length_list = []             # columns의 번호 전달
# 		for i in range( len(ocr_columns) ):
# 			columns_length_list.append(i)
		
# 		easyocr_list = car_ocr.easyocr_list  # easyocr 결과 전달

# 		return render_template('index.html', filename=filename, easyocr_list=easyocr_list, ocr_columns=ocr_columns,
# 								columns_length_list=columns_length_list )     # filename을 index.html에 전달

# 	else:
# 		flash('Allowed image types are -> png, jpg, jpeg, gif')
# 		return redirect(request.url)

# @app.route('/display/<filename>')
# def display_image(filename):
# 	# print('display_image filename: ' + filename)
# 	return redirect(url_for('static', filename='uploads/' + filename), code=301)



# if __name__ == "__main__":
#     app.run(host = "0.0.0.0")   # 127.0.0.1 == localhost








