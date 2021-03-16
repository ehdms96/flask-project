from flask import Flask, render_template, request, send_file
from werkzeug import secure_filename
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN as mt
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
import os
workers = 0 if os.name == 'nt' else 4


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #파일 업로드 용량 제한 단위:바이트

@app.errorhandler(404)
def page_not_found(error):
	app.logger.error(error)
	return render_template('page_not_found.html'), 404

#HTML 렌더링
@app.route('/')
def home_page():
	return render_template('home.html')

#업로드 HTML 렌더링
@app.route('/upload')
def upload_page():
	return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		#저장할 경로 + 파일명
		f.save('./uploads/' + secure_filename(f.filename))
		return render_template('check.html')
	else:
		return render_template('page_not_found.html')

#다운로드 HTML 렌더링
@app.route('/downfile')
def down_page():
	files = os.listdir("./uploads")
	return render_template('filedown.html',files=files)

#파일 다운로드 처리
@app.route('/fileDown', methods = ['GET', 'POST'])
def down_file():
	if request.method == 'POST':
		sw=0
		files = os.listdir("./uploads")
		for x in files:
			if(x==request.form['file']):
				sw=1
				path = "./uploads/" 
				return send_file(path + request.form['file'],
						attachment_filename = request.form['file'],
						as_attachment=True)

		return render_template('page_not_found.html')
	else:
		return render_template('page_not_found.html')

@app.route('/faceNet')
def main():
    faces_dirpath = '/root/uploads/'
    faces_list = os.listdir(faces_dirpath)

    pixels1 = extract_face(faces_dirpath + faces_list[0])
    pixels2 = extract_face(faces_dirpath + faces_list[1])
    mtcnn = MTCNN()
    img1 = Image.open(faces_dirpath + faces_list[0])
    img2 = Image.open(faces_dirpath + faces_list[1])
    img_cropped1 = mtcnn(img1)
    img_cropped2 = mtcnn(img2)
    
    img_embedding1 = resnet(torch.as_tensor(np.asarray(img_cropped1)).unsqueeze(0))
    img_embedding2 = resnet(torch.as_tensor(np.asarray(img_cropped2)).unsqueeze(0))

    dists = (img_embedding1 - img_embedding2).norm().item()
    return 'similarity is ' + dists

def extract_face(filename, required_size=(160, 160)):
    # 파일에서 이미지 불러오기
    image = Image.open(filename)
    # RGB로 변환, 필요시
    image = image.convert('RGB')
    # 배열로 변환
    pixels = np.asarray(image)
    # 감지기 생성, 기본 가중치 이용
    detector = mt()
    # 이미지에서 얼굴 감지
    results = detector.detect_faces(pixels)
    # 첫 번째 얼굴에서 경계 상자 추출
    x1, y1, width1, height1 = results[0]['box']
    # 버그 수정
    x1, y1 = abs(x1), abs(y1)
    x3, y3 = x1 + width1, y1 + height1
    # 얼굴 추출
    face1 = pixels[y1:y3, x1:x3]
    # 모델 사이즈로 픽셀 재조정
    image1 = Image.fromarray(face1)
    image1 = image1.resize(required_size)
    filedir = []
    for i in filename.split('/'):
      filedir.append(i)
    image1.save('/root/uploads/'+filedir[-1])
    face_array1 = np.asarray(image1)
    
    return face_array1

if __name__ == '__main__':
	#서버 실행
	app.run(host='0.0.0.0', debug = True)