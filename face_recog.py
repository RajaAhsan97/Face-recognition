"""
    For detection of Face region in the Image YuNet face detector is used, and for face recognition
    SFace model is used...

    If features of the face image is found in the test image, then it is designated as recognized face.
    for this the features of both images are matched using L2 or Cosine match score

    1. L2 score
            for two identical images the score will be close to 0
    2. Cosine score
            for two identical, images the score will be closer to 1 

    Conclusion:
        feature match using cosine and simple show same score

        *   for higher resolution (149x191) of face region L2 distance score is < 0.05 
"""

import os
import argparse
import cv2
import numpy as np
import time

print("*********************************")
print("Face Recognition using SFace")
print("*********************************")

base_dir = os.path.dirname(__file__)

base_dir_fldr = os.listdir(base_dir)

# get models path 
for directory in base_dir_fldr:
    if directory == "models":
        model_dir_path = os.path.abspath(os.path.join(base_dir, directory))
        model_nm = os.listdir(model_dir_path)
        for model in model_nm:
            model_pth = os.path.join(model_dir_path, model)
            if model == "YuNet":
                YuNet = os.path.join(model_pth, os.listdir(model_pth)[0])
            elif model == "SFace":
                Sface = os.path.join(model_pth, os.listdir(model_pth)[0])


# path to models files
parser = argparse.ArgumentParser()
parser.add_argument("--Face_Detection_YuNet", type=str, default=YuNet, help="Path to face detection model (downloaded from opencvzoo/models/face_detection_yunet)")
parser.add_argument("--Face_Recognition_SFace", type=str, default=Sface, help="Path to face recognition model (downloaded from opencvzoo/models/face_recognition_sface)")
parser.add_argument("--Face_Detection_score_thres", type=float, default=0.85, help="filter the faces with score < score threshold")
parser.add_argument("--Face_Detection_nms_thres", type=float, default=0.3, help="Suppress the k bboxes >= nms threshold")
parser.add_argument("--Face_Detection_Top_k", type=int, default=5000, help="keep k bboxes before nms")
parser.add_argument("--Face_Recognition_cosine_simil_thres", type=float, default=0.8, help="Indicate the detected face if the features of both face images are similar, which is represented by similarity score >0.9")
args = parser.parse_args()


# Initialize YuNet face Detector model instance
facedetector = cv2.FaceDetectorYN_create(args.Face_Detection_YuNet, "", (0,0), args.Face_Detection_score_thres, args.Face_Detection_nms_thres, args.Face_Detection_Top_k)

# Initialize Sface Face Recognizer model instance
facerecognizer = cv2.FaceRecognizerSF.create(args.Face_Recognition_SFace, "")


# load target faces and extract their features using SFace model which will be utilzed to recognize faces in the given sample image
tar_fc_fdr_nm = "target faces"
tar_fc_fdr_pth = os.path.join(base_dir, tar_fc_fdr_nm)

tar_faces = os.listdir(tar_fc_fdr_pth)
target_faces = len(tar_faces)

tar_img_features = {}
for faces in tar_faces:
    image_path = os.path.join(tar_fc_fdr_pth, faces)

    # read image from the path
    image = cv2.imread(image_path)
    image_resol = np.shape(image)

    channels = 1 if len(image_resol) == 2 else image_resol[-1]

    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image_nm = image_path.rsplit("\\")[-1]
    tar_img_features[image_nm] = facerecognizer.feature(image)


# Load sample image in which the faces will be recognized
image2 = cv2.imread("selfie.jpg")
image2_resol = np.shape(image2)

channels = 1 if len(image2_resol) == 2 else image2_resol[-1]

if channels == 1:
    image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif channels == 4:
    image2 = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
# set input image size for model (W, H)
facedetector.setInputSize((image2_resol[1], image2_resol[0]))

# detect faces from the sample image
face_detect_strt_tm = time.time()
_, faces = facedetector.detect(image2)
face_detect_end_tm = time.time()

faces = faces if faces is not None else []

img2 = image2.copy()
tar_img_feat = tar_img_features.copy()
feat_match_cnt = 0

#--------------------
Feat_extract_tm = {}
Feat_mtch_tm = {}
cnt = 1
# ------------------

# for the detected faces extract the features match them with the target faces features, if match score >0.9 then the face as recognized
# recognized faces are enclosed in green boxes
faces_rec_start_tm = time.time()
for face in faces:
    fc, confidence_score = list(face[:-1].astype('i')), str(np.round(face[-1], 3))

    cv2.rectangle(image2, (fc[0], fc[1]), (fc[0]+fc[2], fc[1]+fc[3]), (0,0,255), 2)

    #aligned_face = facerecognizer.alignCrop(image2, face)
    feat_extrt_strt_tm = time.time()
    features = facerecognizer.feature(img2[fc[1]:fc[1]+fc[3], fc[0]:fc[0]+fc[2]])
    feat_extrt_end_tm = time.time()

    # -------------------------
    Feat_extract_tm[cnt] = np.round(feat_extrt_end_tm - feat_extrt_strt_tm, 4)
    # -------------------------

    feat_mtch_strt_tm = time.time()
    for k in tar_img_feat.keys():
        tar_features = tar_img_feat[k]
        cosine_score = facerecognizer.match(features, tar_features, cv2.FaceRecognizerSF_FR_COSINE)

        if cosine_score >= args.Face_Recognition_cosine_simil_thres:
            cv2.rectangle(image2, (fc[0], fc[1]), (fc[0]+fc[2], fc[1]+fc[3]), (0,255,0), 2)
            cv2.putText(image2, confidence_score, (fc[0],fc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)        
            tar_img_feat.pop(k)
            feat_match_cnt += 1
            break
    feat_mtch_end_tm = time.time()

    # -------------------------
    Feat_mtch_tm[cnt] = np.round(feat_mtch_end_tm - feat_mtch_strt_tm, 4)
    cnt += 1
    # -------------------------
    
faces_rec_end_tm = time.time()

faces_detect_time = np.round(face_detect_end_tm - face_detect_strt_tm, 4)
faces_recog_time = np.round(faces_rec_end_tm - faces_rec_start_tm, 4)
print("------------------------------------------")
print("Face detection time (sec): ", faces_detect_time)
print("Face recognition time (sec): ", faces_recog_time)
print("******************************************")
print("Target faces: ", target_faces)
print("Faces Recognized: ", feat_match_cnt)
print("------------------------------------------")

cv2.imwrite("face recognized.jpg", image2)
