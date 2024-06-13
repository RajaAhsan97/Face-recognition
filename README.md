This repository contains different techniques for face recognition on the largest selfie image. Prior to face recognition faces in the image are detected using YuNet face detector - a deep
learning neural network model. Faces in the image are recognized by matching the features of detected faces with the target faces. For feature match DNN model and statistical methods are used 
as mentioned below:

1. SFace
2. SSIM
3. SIFT

Where the SSIM and SIFT are the statistical methods.

.py file description

i.   Face recognition using SFace  ----  face_recog.py
ii.  Face recognition using SSIM   ----  face_recog1.py
iii. Face recognition using SIFT   ----  face_recog2.py


Note: 
      i.    For SFace the similarity between two images is measured by computing the cosine distance, therefore for a match I have set the score threshold to 0.8. For a match the computed cosine 
            distance score > score threshold.  
      ii.   For SSIM the two images which are to be compared must exhibit the same resolution. Therefore for the detected face image resolution I have grabbed the target faces whose resolution 
            matches with detected faces and then computed the similarity measure between the images. For a match between two images I have set the SSIM score threshold to 0.8, if the computed score 
            > SSIM score threshold then the detected faces is marked as recognized.
      iii.  For SIFT the similarity between two images is measured by computing the L2_Norm distances between the common features in the two images. For a perfect feature match the distance is 
            small and for the two images perfect match features are further filtered by using LOWE distance ratio criteria (ref: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html).


largest selfie image
![selfie](https://github.com/RajaAhsan97/Face-recognition/assets/155144523/742abbb0-86d2-4756-b4ed-9f21adf3dfc2)

      
