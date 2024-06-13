"""
    For detection of Face region in the Image YuNet face detector is used, and for face recognition
    SSIM is used which compare the features of the target face image with the detected faces. For perfect
    match of features of the two images the score will be 1. Thus, I have defined the SSIM score threshold to
    0.8, if the computed score of the target and the detected face region is > SSIM score threshold then the detected face is
    designated as the recognized.
"""

import os
import argparse
import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as SSIM

print("*********************************")
print("Face Recognition using SSIM Index")
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
parser.add_argument("--Face_Detection_score_thres", type=float, default=0.85, help="filter the faces with score < score threshold")
parser.add_argument("--Face_Detection_nms_thres", type=float, default=0.3, help="Suppress the k bboxes >= nms threshold")
parser.add_argument("--Face_Detection_Top_k", type=int, default=5000, help="keep k bboxes before nms")
parser.add_argument("--SSIM_score_threshold", type=float, default=0.8, help="Indicate recognized faces with SSIM score > SSIM_score_threshold")
args = parser.parse_args()


# Initialize YuNet face Detector model instance
facedetector = cv2.FaceDetectorYN_create(args.Face_Detection_YuNet, "", (0,0), args.Face_Detection_score_thres, args.Face_Detection_nms_thres, args.Face_Detection_Top_k)


# load target faces and extract their features using SFace model which will be utilzed to recognize faces in the given sample image
tar_fc_fdr_nm = "target faces"
tar_fc_fdr_pth = os.path.join(base_dir, tar_fc_fdr_nm)

tar_faces = os.listdir(tar_fc_fdr_pth)
target_faces = len(tar_faces)

tar_imgs = {}
for faces in tar_faces:
    image_path = os.path.join(tar_fc_fdr_pth, faces)

    # read image from the path
    image = cv2.imread(image_path)
    image_resol = np.shape(image)

    channels = 1 if len(image_resol) == 2 else image_resol[-1]

    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    image_nm = image_path.rsplit("\\")[-1]
    tar_imgs[image_nm] = image


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
tar_images = tar_imgs.copy()
feat_match_cnt = 0


"""
    For SSIM the dimensions of the two images must be same, in my case the resolutions of all target and detected face images are
    different. Thus to compute the similarity score of two same resoultion images, I have extracted the target images whose resolution
    matches with the detected face image and then used the extracted images for similarity measure with the detected face region.
"""
faces_rec_start_tm = time.time()
for face in faces:
    fc, confidence_score = list(face[:-1].astype('i')), str(np.round(face[-1], 3))

    cv2.rectangle(image2, (fc[0], fc[1]), (fc[0]+fc[2], fc[1]+fc[3]), (0,0,255), 2)

    crop_face = cv2.cvtColor(img2[fc[1]:fc[1]+fc[3], fc[0]:fc[0]+fc[2]], cv2.COLOR_BGR2GRAY)

    match_faces_res = {}
    for k in tar_images.keys():
        if np.shape(tar_images[k]) == np.shape(crop_face):
            match_faces_res[k] = tar_images[k]

    for KEY in match_faces_res.keys():
        tar_image = match_faces_res[KEY]
        SSIM_score = SSIM(crop_face, tar_image)
        if SSIM_score > args.SSIM_score_threshold:        
            cv2.rectangle(image2, (fc[0], fc[1]), (fc[0]+fc[2], fc[1]+fc[3]), (0,255,0), 2)
            cv2.putText(image2, confidence_score, (fc[0],fc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)        
            tar_images.pop(KEY)
            feat_match_cnt += 1
            match_faces_res = {}   
            break
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

cv2.imwrite("face recognized_SSIM.jpg", image2)
