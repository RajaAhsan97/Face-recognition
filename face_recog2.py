import csv
import os
import numpy as np
import argparse
import cv2


base_dir = os.path.dirname(__file__)

model_dir = os.path.join(base_dir, "models", "YuNet")

model_path = os.path.join(model_dir, os.listdir(model_dir)[0])

#----------------------------------------
sift_face_match_pth = os.path.join(base_dir, "SIFT faces match")
#----------------------------------------

#----------------------------------------
# set full path for target faces images 
tar_faces_dir = "target faces"
tar_faces_path = os.path.join(base_dir, tar_faces_dir)
target_faces = os.listdir(tar_faces_path)

for tar_face in target_faces:
    target_faces[target_faces.index(tar_face)] = os.path.abspath(os.path.join(tar_faces_path, tar_face))
#----------------------------------------

#----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--Face_Detector_YuNet_model", type=str, default=model_path, help="Path for YuNet model (downloaded from opencv zoo)")
parser.add_argument("--Face_Detector_score_thres", type=float, default=0.85, help="filter boxes with score < score_threshold")
parser.add_argument("--Face_Detector_nms_thres", type=float, default=0.3, help="Suppress boxes >= nms_threshold")
parser.add_argument("--Face_Detector_K_boxes", type=int, default=5000, help="keep k bboxes before nms")
parser.add_argument("--Image_Feature_match_distance", type=int, default=100, help="Designate face as recognized if the distance between to faces keypoint descriptors is < Image_Features_match_distance")
parser.add_argument("--D_LOWE_distance_ratio_thres", type=float, default=0.7, help="SIFT images feature match distance ratio threshold propose by LOWE")
args = parser.parse_args()
#----------------------------------------


#----------------------------------------
facedetector = cv2.FaceDetectorYN_create(args.Face_Detector_YuNet_model, "", (0,0), args.Face_Detector_score_thres, args.Face_Detector_nms_thres, args.Face_Detector_K_boxes)


# load Image
image = cv2.imread("selfie.jpg")
image_res = np.shape(image)
channels = 1 if len(image_res) == 2 else image_res[-1]

# YuNet face detector requires RGB image, so i have converted the input image channels to RGB
if channels == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif channels == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

facedetector.setInputSize((image_res[1], image_res[0]))

_, faces = facedetector.detect(image)

faces = faces if faces is not None else []

# Create sift object instance
sift = cv2.SIFT_create()

# initialize feature match object instance
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

match_distance = []
out_image = image.copy()
Faces_recognized = 0
for face in faces:
    fc, conf_score = list(face[:-1].astype('i')), str(face[-1])

    x1, y1 = fc[0], fc[1]
    x2, y2 = fc[0]+fc[2], fc[1]+fc[3]

    cv2.rectangle(out_image, (x1,y1), (x2,y2), (0, 0, 255), 2)
    
    # crop the face region from the image
    crp_face = image[y1:y2, x1:x2]
    # convert crop face color channel to Gray level
    crp_face_gray = cv2.cvtColor(crp_face, cv2.COLOR_BGR2GRAY)

    # extract keypoints in crop face
    kp = sift.detect(crp_face_gray, None)

    # compute descriptors from keypoints
    kp, des = sift.compute(crp_face_gray, kp)

    # read target images and extract features (SIFT)
    for tar_F in target_faces:
        #------------------------------
        # target face
        tar_img = cv2.imread(tar_F)
        tar_img_gray = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)

        tar_kp = sift.detect(tar_img_gray, None)
        tar_kp, tar_des = sift.compute(tar_img_gray, tar_kp)
        #------------------------------

        # Match features of the detected face with the target faces
        knn_matches = matcher.knnMatch(des, tar_des, 2)

        # D Lowe distance ratio for filtering good feature matches between two images 
        good_match_dist = []
        for m, n in knn_matches:
            if m.distance < args.D_LOWE_distance_ratio_thres * n.distance:
                good_match_dist.append(m.distance)

        match_dist = np.sum(good_match_dist)/len(good_match_dist)

        # for two similar faces images the match distance is less as compared to two different faces images
        if match_dist < args.Image_Feature_match_distance:        
            cv2.rectangle(out_image, (x1,y1), (x2,y2), (0,255,0), 2)
            result = cv2.drawMatchesKnn(tar_img_gray, tar_kp, crp_face_gray, kp, knn_matches, None)
            cv2.imwrite(os.path.join(sift_face_match_pth, tar_F.split("\\")[-1]), result)
            match_distance.append(match_dist)
            target_faces.remove(tar_F)
            Faces_recognized += 1
            break
    

cv2.imwrite("face recognizes_SIFT.jpg", out_image)

print("Faces Detected: ", len(faces))
print("Faces Recognized: ", Faces_recognized)
