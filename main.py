import os
import cv2


sample= cv2.imread("fingerprint2c.jpg")
score = 0
kp1,kp2,mp = None,None, None

fp_image = cv2.imread("fingerprint2.jpg")
sift = cv2.SIFT_create()                              # scale invariant feature transform (allows to extract key points from individual images)
keypoints_1,descriptor_1 = sift.detectAndCompute(sample,None)
keypoints_2,descriptor_2 = sift.detectAndCompute(fp_image,None)
matches =  cv2.FlannBasedMatcher({'algorithm':1,'trees':10},{}).knnMatch(descriptor_1,descriptor_2,k=2)
    
match_point =[]
for p,q in matches:
    if p.distance <0.1*q.distance:
        match_point.append(p)
keypoints =  0
if len(keypoints_1)<len(keypoints_2):
    keypoints = len(keypoints_1)
else:
    keypoints = len(keypoints_2)
    
if len(match_point)/keypoints *100 > score:
    score = len(match_point)/keypoints*100
    image = fp_image
    kp1,kp2,mp = keypoints_1,keypoints_2,match_point
print('match:{}%'.format(score))
result = cv2.drawMatches(sample,kp1,fp_image,kp2,mp,None)
result = cv2.resize(result,None,fx=1,fy=1)
cv2.imshow('finger_print_match',result)
cv2.waitKey(0)
