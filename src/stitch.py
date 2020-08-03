# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:42:59 2020

@author: Maruthi Reddy
"""
import cv2
import copy
import numpy as np
import random
import os
import sys
import glob
import argparse
from scipy.spatial.distance import cdist


def matching_points_imgs(img1,img2):
    '''
    

    Parameters
    ----------
    img1 : image 1.
    img2 : image2.

    Returns
    -------
    matches : matched key points of two images.
    match_kp1 : keypoints of image1.
    match_kp2 : keypoints of image2.

    '''
    features_sift=cv2.xfeatures2d.SIFT_create()
    kp1,des1=features_sift.detectAndCompute(img1,None)
    kp2,des2=features_sift.detectAndCompute(img2,None)
    
    pairwiseDistances = cdist(des1, des2, 'sqeuclidean')
    threshold =7000
    
    index_matchkp1 = np.where(pairwiseDistances < threshold)[0]
    index_matchkp2 = np.where(pairwiseDistances < threshold)[1]

    match_kp1 = np.array([kp1[point].pt for point in index_matchkp1])
    match_kp2 = np.array([kp2[point].pt for point in index_matchkp2])
    matches=np.concatenate((match_kp1, match_kp2), axis=1)
    return matches,match_kp1,match_kp2

def ransac(matchingKeyPoints,keypoints_1,keypoints_2,threshold=0.5,N=1000):
    '''
    

    Parameters
    ----------
    matches : matched key points of two images.
    match_kp1 : keypoints of image1.
    match_kp2 : keypoints of image2.
    threshold : threshold lessthan this error considered as good point, optional
        DESCRIPTION. The default is 0.5.
    N : the number of iteration the RANSAC should run to obtain a good approximation
        DESCRIPTION. The default is 1000.

    Returns
    -------
    H : Homography matrix which transforms using this we transform points on img1 to img2.

    '''
    r  = random.SystemRandom()
    current_val=0
    H=[]

    
    for iterations in range(N):
        p1 = r.choice(matchingKeyPoints)
        p2 = r.choice(matchingKeyPoints)
        p3 = r.choice(matchingKeyPoints)
        p4 = r.choice(matchingKeyPoints)
        
        fourPairs = np.concatenate(([p1],[p2],[p3],[p4]),axis=0)
        src_points = np.float32(fourPairs[:,0:2])
        dst_points = np.float32(fourPairs[:,2:4])
        
        H_curr = cv2.getPerspectiveTransform(src_points, dst_points)
        rank_H=np.linalg.matrix_rank(H_curr)
        
        if rank_H<3:
            continue
        
        
        key_1=np.ones((keypoints_1.shape[0],keypoints_1.shape[1]+1))
        key_1[:,:-1]=keypoints_1
        
        points_img2=np.zeros((keypoints_2.shape))
        
        for i,x in enumerate(key_1):
            y=np.matmul(H_curr,x)
            points_img2[i]=(y/y[2])[0:2]
        
        #print(points_img2.shape)    
        error=np.linalg.norm(keypoints_2 - points_img2,axis=1)**2
        #print(error)
        index=np.where(error<threshold)[0]
        inliers=matchingKeyPoints[index]
        #print('len of inliers',len(inliers))
        if len(inliers)>current_val:
            current_val=len(inliers)
            H = H_curr.copy()
    return H

# def trim(img):
    
#     if not np.sum(img[:,0]):
#         return trim(img[:,1:])

#     if not np.sum(img[:,-1]):
#         return trim(img[:,:-2])
    
#     if not np.sum(img[0]):
#         return trim(img[1:])
   
#     if not np.sum(img[-1]):
#         return trim(img[:-2])
    
#     return img


def trim(image):
    '''
    

    Parameters
    ----------
    image : image after stitching.

    Returns
    image after removing black pixels

    '''
    
    black = np.zeros(3)
    x = 0
    y = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            if not np.array_equal(pixel, black):
                if j > x:
                   x = j
                if i > y:
                   y = i
    return image[0:y,0:x]

    
   
def stitching_imges(img1,img2,H):
    '''
    

    Parameters
    ----------
    img1 : image1
    img2 : image2.
    H : homography matrix

    Returns
    -------
    stitched_img : stitches img1 and img2 and return stitched image 

    '''
    
    # final_img = cv2.warpPerspective(img2,H,(img1.shape[1] + img2.shape[1] , img1.shape[0]+img2.shape[0]))
    # final_img[0:img1.shape[0],0:img1.shape[1]] = img1
    # return final_img
    stitched_img = cv2.warpPerspective(img2, H,(int(img2.shape[1] + img1.shape[1]),int(img2.shape[0] + img1.shape[0] )))
                                        
    stitched_img[0:img1.shape[0], 0:img1.shape[1]] = img1

    return stitched_img



    

def main():
    """
         it reads all images from a directory to a list
         as color images and gray scale images.
         we cannot create panorama if there is one or less images.
         we take one image from the list of images and find matching points with remaining images .
         we find the images with maximum number of matched points and try to stitch two images in both 
         forward and backward order.
         out of two panaromas we take the image with less number of black pixels.
         we remove the two images used for creating panorama
         we add stitchd images to image list and repeat the process.
         after completion we write image to same folder
         
    """

    if len(sys.argv) != 2:
        print("Invalid number of arguments.")
        return

    directory = str(sys.argv[1])
    
    image_Directory = os.path.join(directory, '*.jpg')
    #if there is existing panorama removing it
    if os.path.exists(os.path.join(directory, 'panorama.jpg')):
        os.remove(os.path.join(directory, 'panorama.jpg'))
    colorImages = [cv2.imread(file) for file in glob.glob(image_Directory)]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_Directory)]
    
    if len(images)==0 or len(images)==1:
        print('You need to have morethan one image to create panorama')
    else:
        n=len(images)
        count=0
        while((len(images))!=1):
            list_matches_index1=[]
            list_matches1=[]
            list_matches2=[]
            
            for i in range(1,len(images)):
                m1,k1,k2= matching_points_imgs(images[0],images[i])
                m2,k11,k21=matching_points_imgs(images[i],images[0])
                list_matches_index1.append(m1.shape[0])
                list_matches1.append([m1,k1,k2])
                list_matches2.append([m2,k11,k21])
                
            max_index=list_matches_index1.index(max(list_matches_index1))
            matches,keypoints1,keypoints2=list_matches1[max_index]
            matches1,keypoints11,keypoints21=list_matches2[max_index]
            
            H01=ransac(matches,keypoints1,keypoints2)
            H10=ransac(matches1,keypoints11,keypoints21)
            P01=stitching_imges(colorImages[max_index+1],colorImages[0],H01)
            P02=stitching_imges(colorImages[0],colorImages[max_index+1],H10)
            P1=cv2.cvtColor(P01, cv2.COLOR_BGR2GRAY)
            P2=cv2.cvtColor(P02, cv2.COLOR_BGR2GRAY)
            c1=cv2.countNonZero(P1)
            c2=cv2.countNonZero(P2)
            if c2>c1:
                P1=P2
                P01=P02
            
            P01=trim(P01)
            P1=trim(P1)
            images.pop(0)
            images.pop(max_index)
            colorImages.pop(0)
            colorImages.pop(max_index)
            colorImages.append(P01)            
            images.append(P1)
            count+=1
            if (count>n):
                print('the system ran into error taking long time')
                break
        cv2.imwrite(os.path.join(directory, 'panorama.jpg'), colorImages[0])
        print('panorama created successfully') 
        
if __name__ == "__main__":
    main()            