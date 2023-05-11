import mxnet

import cv2
import numpy as np
import os
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
from numpy import dot
from numpy.linalg import norm
import xlsxwriter
import math
import face_model_mxnet

from sklearn.preprocessing import normalize
from scipy import spatial

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
#parser.add_argument('--model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--model', default='./model/Arcface/model,0',help='path to load model.')
#parser.add_argument('--ga-model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--ga-model', default='./model/Arcface/model,0',help='path to load model.')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--MaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop',help='')
parser.add_argument('--WoMaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop',help='')
parser.add_argument('--SavePath', default='./feature/',help='path to save feature txt file.')
parser.add_argument('--FeaturePath', default='',help='path to load feature txt file.')
parser.add_argument('--FeaturePath2', default='',help='path to load feature txt file.')
# parser.add_argument('--Protocal', default='./Protocol_verification/AFDB_new.csv',help='')
parser.add_argument('--Protocal', default='./Protocol_verification/AFDB_final_Jeff.csv',help='')
# parser.add_argument('--Method', default=['Arcface'],help='Model to test.')
parser.add_argument('--Method', default=['Arcface','Cosface','Sphereface','Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
# parser.add_argument('--Method', default=['Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
args = parser.parse_args()


#args.model = './model/Arcface/model,0'

model = face_model_mxnet.FaceModelMxnet(args)
face_name = "pedro1.jpg"

face_img = model.read(face_name)
aligned = model.get_input(face_img)
print(type(aligned[0]))
print(aligned[0].shape)
embedding1 = model.get_feature(aligned[0])
#print(embedding1)

face_name = "pedro2.jpg"
face_img = cv2.imread(face_name)
aligned = model.get_input(face_img)
print(type(aligned[0]))
print(aligned[0].shape)
print(aligned[0])
embedding2 = model.get_feature(aligned[0])
#print(embedding2)


dot = np.sum(np.multiply([embedding1], [embedding2]), axis=1)
norm = np.linalg.norm([embedding1], axis=1) * np.linalg.norm([embedding2], axis=1)
sim = dot / norm
dist = np.arccos(sim) / math.pi
sim2 = 1 - spatial.distance.cosine(embedding1, embedding2) 
print(sim2)

