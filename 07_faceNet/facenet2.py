import tensorflow as tf
import numpy as np
from mtcnn import MTCNN as mt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as pp
import os
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def face():
    faces_dirpath = '/home/ubuntu/flask-project/07_faceNet/uploads/faces/'
    faces_list = os.listdir(faces_dirpath)
    num = len(faces_list)
    myface_dirpath = '/home/ubuntu/flask-project/07_faceNet/uploads/myface/'
    my_face = os.listdir(myface_dirpath)
    min = 0x7fffffff
    index = 0;
    required_size=(160, 160)
    for i in range(num):
        print(i)
        image1 = Image.open(faces_dirpath + faces_list[i])
        image2 = Image.open(myface_dirpath + my_face[0])
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        pixels1 = np.asarray(image1)
        pixels2 = np.asarray(image2)
        detector = mt()
        results1 = detector.detect_faces(pixels1)
        results2 = detector.detect_faces(pixels2)
        x1, y1, width1, height1 = results1[0]['box']
        x2, y2, width2, height2 = results2[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = abs(x2), abs(y2)
        x3, y3 = x1 + width1, y1 + height1
        x4, y4 = x2 + width2, y2 + height2
        face1 = pixels1[y1:y3, x1:x3]
        face2 = pixels2[y2:y4, x2:x4]
        image1 = Image.fromarray(face1)
        image2 = Image.fromarray(face2)
        image1 = image1.resize(required_size)
        image2 = image2.resize(required_size)
        image1.save(faces_dirpath + faces_list[0])
        image2.save(faces_dirpath + faces_list[1])

        pxl1 = np.asarray(image1)
        pxl2 = np.asarray(image2)
        mtcnn = MTCNN()
        img1 = Image.open(faces_dirpath + faces_list[0])
        img2 = Image.open(faces_dirpath + faces_list[1])
        img_cropped1 = mtcnn(img1)
        img_cropped2 = mtcnn(img2)
        img_embedding1 = resnet(torch.as_tensor(np.asarray(img_cropped1)).unsqueeze(0))
        img_embedding2 = resnet(torch.as_tensor(np.asarray(img_cropped2)).unsqueeze(0))
        dist = (img_embedding1 - img_embedding2).norm().item()
        if min > dist:
            min = dist
            index = i

        #result_face = Image.open(faces_dirpath + faces_list[i])
        #pp.imshow(result_face)
    #print(faces_list[i])
    return faces_list[i]

#face()
