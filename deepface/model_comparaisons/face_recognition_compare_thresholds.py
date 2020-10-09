# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:28:45 2020

@author: aktas
"""

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.commons import functions, realtime, distance as dst

import time
import os


base_path = 'FaceRecognition_dataset3\\'
osym_path = 'osym\\'

#models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
model_name= "Facenet"  # "Facenet"
distance_metric = "euclidean_l2" # "euclidean_l2"
threshold = 1.05
detector = "ssd" #opencv

log_file =  open("model_performation_logs_threshold.txt", "w+")


if model_name == 'VGG-Face':
    log_file.write("------ Using VGG-Face model ------ \n\n\n")
    model = VGGFace.loadModel()
elif model_name == 'Facenet':
    log_file.write("------ Using Facenet model ------ \n\n\n")
    model = Facenet.loadModel()
    
input_shape = model.layers[0].input_shape
if type(input_shape) == list:
    input_shape = input_shape[0][1:3]
else:
    input_shape = input_shape[1:3]
      
input_shape_x = input_shape[0]
input_shape_y = input_shape[1]    
    
    
#accuracy calculation for positive samples
log_file.write("Accuracy Calculation for Positive Samples \n")
    
true_count = 0
total_sample_count = 0
no_face_count = 0
for imageName in os.listdir(base_path + osym_path):
    input_folder = base_path + imageName.split('_osym')[0]
    for sampleImage in os.listdir(input_folder+'\\'):
        print(sampleImage)
        input_foto_path = input_folder +'\\' + sampleImage
        osym_foto_path = base_path + osym_path + imageName
        #----------------------

        img1, face_flag = functions.preprocess_face(img=osym_foto_path, target_size=(input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector)
        if face_flag == False:
            print("no face detected on" + osym_foto_path + "\n")
            log_file.write("no face detected on" + osym_foto_path + "\n")
            no_face_count += 1
            continue
        img2,face_flag = functions.preprocess_face(img=input_foto_path, target_size=(input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector)
        if face_flag == False:
            print("no face detected on" + input_foto_path + "\n")
            log_file.write("no face detected on" + input_foto_path + "\n")
            no_face_count += 1
            continue
            #----------------------
            #find embeddings

        img1_representation = model.predict(img1)[0,:]
        img2_representation = model.predict(img2)[0,:]
                     
        #--------------------
        #distance calculation
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)) 
        #----------------------
        #decision
        if distance <= threshold:
            identified =  "true"
            true_count += 1                    
        else:
            identified =  "false"
        total_sample_count += 1    

        log_file.write(" image " + sampleImage + " verified " + identified + " distance " + str(distance) + " max_threshold_to_verify " + str(threshold) + "\n")
                
log_file.write("total positive sample image with face detected " + str(total_sample_count) + "accuracy is %" + str((true_count/total_sample_count)*100) + "\n")
            
    
#accuracy calculation for negative samples
log_file.write("Accuracy Calculation for Negative Samples  \n")
    
true_count = 0
total_sample_count = 0
no_face_count = 0
log_file.write("Using " + detector + " backend \n")

for imageName in os.listdir(base_path + osym_path):
    input_folder = base_path + "not_" + imageName.split('_osym')[0]
    for sampleImage in os.listdir(input_folder+'\\'):
        print(sampleImage)
        input_foto_path = input_folder +'\\' + sampleImage
        osym_foto_path = base_path + osym_path + imageName
        #----------------------

        img1,face_flag = functions.preprocess_face(img=osym_foto_path, target_size=(input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector)
        if face_flag == False:
            print("no face detected on" + osym_foto_path + "\n")
            log_file.write("no face detected on" + osym_foto_path + "\n")
            no_face_count += 1
            continue
        img2,face_flag = functions.preprocess_face(img=input_foto_path, target_size=(input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector)
        if face_flag == False:
            print("no face detected on" + input_foto_path + "\n")
            log_file.write("no face detected on" + input_foto_path + "\n")
            no_face_count += 1
            continue
                    
        #----------------------
        #find embeddings

        img1_representation = model.predict(img1)[0,:]
        img2_representation = model.predict(img2)[0,:]
                     
        #--------------------
        #distance calculation
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)) 
        #----------------------
        #decision
        if distance <= threshold:
            identified =  "true"
        else:
            identified =  "false"
            true_count += 1 
        total_sample_count += 1    

        log_file.write(" image " + sampleImage + " verified " + identified + " distance " + str(distance) + " max_threshold_to_verify " + str(threshold) + "\n")
                
log_file.write("total negative sample image with face detected " + str(total_sample_count) + "accuracy is %" + str((true_count/total_sample_count)*100) + "\n")
    
log_file.close() 
        