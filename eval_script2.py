from os import device_encoding
import sys 
import torch
import imageio
import torch.nn as nn
import cv2
from tqdm import tqdm

import numpy as np
import os
from sklearn.preprocessing import normalize
import argparse

from collections import OrderedDict
import math
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import xlsxwriter
#import face_model_mxnet

from face_models.iresnet import iresnet100

#import loader
#import model

parser = argparse.ArgumentParser(description='OCFR script')

#parser.add_argument('--device', default='gpu', help='gpu id')
parser.add_argument('--model_path', default='model_path', help='path to pretrained model')
parser.add_argument('--image_list', type=str, default='',help='pairs file')

parser.add_argument('--list_pairs', type=str, default='',help='pairs file')
parser.add_argument('--data_path', type=str, default='',help='root path to data')

parser.add_argument('--save_path', type=str, default='',help='root path to data')

# Face model args
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/Arcface/model,0',help='path to load model.')
parser.add_argument('--ga-model', default='./model/Arcface/model,0',help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')


class FaceModel():
    def __init__(self, model_path, save_path,image_list,data_path,ctx_id,args):
        self.gpu_id=ctx_id
        self.model=self._get_model(args)
        #self.detector = self.get_detector(args)
        self.save_path=save_path
        if not(os.path.isdir(self.save_path)):
            os.makedirs(save_path)
        self.image_list=image_list
        self.data_path=data_path

    def _get_model(self, args): 
        model = iresnet100(num_features=512).to(int(self.gpu_id))
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(int(self.gpu_id))))         
        model = model.to(int(self.gpu_id))
        model.train(False)
        return model

    #def get_detector(self,args):
    #    detector = face_model_mxnet.FaceModelMxnet(args) 
    #    return detector


    def _getFeatureBlob(self,input_blob):
        feature_blob = []
        for i in input_blob:
            net_out: torch.Tensor = self.model(i.to(int(self.gpu_id)))
            _embeddings = net_out.detach().cpu().numpy()
            feature_blob.append(_embeddings[0])
        return feature_blob

    def read(self,image_path):
        image = cv2.imread(image_path)
        return image

    def save(self,features,image_path_list,alignment_results):
        # Save embedding as numpy to disk in save_path folder
        for i in tqdm(range(len(features))):
            filename = str(str(image_path_list[i]).split("/")[-1].split(".")[0])
            np.save(os.path.join(self.save_path, filename), features[i])
        np.save(os.path.join(self.save_path,"alignment_results.txt"),np.asarray(alignment_results))

    def process(self,image,bbox): 
        #image_r, al_r = self.detector.get_input(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(112,112))
        image = np.transpose(image, (2,0,1))
        al_r = "N"
        image_r = np.asarray([image], dtype="float32")
        image_r = ((image_r / 255) - 0.5) / 0.5
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        image_r = torch.from_numpy(image_r)
        print(image_r.shape)
        return image_r,al_r

    def distance(self,embedding1, embedding2):
        dot = np.sum(np.multiply([embedding1], [embedding2]), axis=1)
        norm = np.linalg.norm([embedding1], axis=1) * np.linalg.norm([embedding2], axis=1)
        sim = dot / norm
        dist = np.arccos(sim) / math.pi
        sim2 = 1 - spatial.distance.cosine(embedding1, embedding2) 
        return sim

    def save_score(self,score):
        with open(os.path.join(self.save_path,"scores.txt"), "ab") as f:
            print(score)
            np.savetxt(f, score)

    def comparison(self,list_pairs):
        with open(list_pairs, "r") as f:
            for line in f:
                emb1=np.load(os.path.join(self.save_path,line.split()[0]+".npy"))
                emb2=np.load(os.path.join(self.save_path,line.split()[1]+".npy")) 
                score=self.distance(emb1,emb2)
                self.save_score(score)

    def read_img_path_list(self):
        with open(self.image_list, "r") as f:
            lines = f.readlines()
            file_path = [os.path.join(self.data_path, line.rstrip().split()[0]) for line in lines]
            bbx = [line.rstrip().split()[1:] for line in lines]
        return file_path ,bbx


    def get_batch_feature(self, image_path_list, bbx, batch_size=64, flip=0):

        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        alignment_results = []
        for i in range(0, len(image_path_list), batch_size):

            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
                tmp_list_bbx = bbx[i : i+batch_size]

            else:
                tmp_list = image_path_list[i :]
                tmp_list_bbx = bbx[i :]

            count += 1

            images = []
            for i  in range(len(tmp_list)):
                image_path=tmp_list[i]
                bbox=tmp_list_bbx[i]

                image=self.read(image_path)

                image,alignment_result=self.process(image,bbox)
                alignment_results.append(alignment_result)
                images.append(image)

            input_blob = images
            emb = self._getFeatureBlob(input_blob)
            features.append(emb)
        features = np.vstack(features) 
        self.save(features,image_path_list,alignment_results)
        return

def main(args):
    model=FaceModel(args.model_path,args.save_path,args.image_list,args.data_path,args.gpu, args)
    file_path, bbx= model.read_img_path_list()
    model.get_batch_feature(file_path, bbx)
    model.comparison(args.list_pairs)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


