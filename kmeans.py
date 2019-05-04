# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:33:00 2019

@author: Sailalitha
@ k means
"""

from PIL import Image
import numpy as np
import pandas as pd
import sys

image = Image.open('cat.png')
image

if len(sys.argv)< 4 : 
    print (" need more arguments")
    sys.exit()
else :
     k = int(sys.argv[1])
     if k<3 :
         print(" need more clusters, increase number")
     inputImageFile= sys.argv[2]
     outputImageFile = sys.argv[3]

image = Image.open(inputImageFile)

# getting the rgb and creating feature vector space
pix_val = np.asarray(image)
r_mean , r_std = np.mean(pix_val[:,0]), np.std(pix_val[:,0])
g_mean , g_std = np.mean(pix_val[:,1]), np.std(pix_val[:,1])
b_mean , b_std = np.mean(pix_val[:,2]), np.std(pix_val[:,2])                                                                                          
w , h = image.size
idx=[]
f_space=[]
x = np.arange(0,w,1)
y = np.arange(0,h, 1)
for i in range(0,h):
    for j in range(0,w):
        f_space.append((pix_val[i][j][0],pix_val[i][j][1],pix_val[i][j][2] , i ,j ))



f_space= np.asarray(f_space)
f = np.zeros((f_space.shape))
feature_nn = f_space
f[:,0] = (f_space[:,0] - np.mean(f_space[:,0]))/ np.std(f_space[:,0])
f[:,1] = (f_space[:,1]- np.mean(f_space[:,1]))/ np.std(f_space[:,1])
f[:,2] = (f_space[:,2] - np.mean(f_space[:,2]))/ np.std(f_space[:,2])
f[:,3]  = (f_space[:,3] - np.mean(f_space[:,3]))/ np.std(f_space[:,3])
f[:,4] = ( f_space[:,4] - np.mean(f_space[:,4]))/ np.std(f_space[:,4])

f

#####

# class of Kmeans 
class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, feature):
         # pass an numpy array into fit
        x,y =feature.shape
        self.centroid = {}
         # assign centroids
        for i in range(0, self.k):
            
            rnd = np.random.randint(0,x)
            self.centroid[i] = feature[rnd] # random initialzation for center
        
      
            
# ## iterate over whole data point to make error 0 for centroid distance.
        for i  in range(0,self.max_iter):
            self.cls = {}
            for i in range(self.k):
                self.cls[i] = [] # 3 clusters inittialized
        
            for points in feature:
                centre_move = []
                for c in self.centroid:
                    x = np.linalg.norm(points - self.centroid[c])
                    centre_move.append(x)
                 
                idx = min(centre_move)
                class_idx = centre_move.index(idx)
                self.cls[class_idx].append(points)
                
                 # once we know which cluster data point belongs to 
                 # add to the class where each idx has a list
            prev_centroid = dict(self.centroid)
 #             print(type(prev_centroid))
            
             # now form the new centroids
            
            for idx in self.cls:
                self.centroid[idx] = np.average(self.cls[idx], axis = 0)
              
            flag = True 
             # now check for the movement of each centroid within tol
            for no in self.centroid:
                sum_tol = np.sum((self.centroid[no]-prev_centroid[no])/prev_centroid[no]*100.0)
 #                 print(sum_tol)
                if  sum_tol > self.tol :
                     flag = False
                     continue 
            if flag ==True : 
                self.final_centroids = self.centroid
 #                 print('done')
                break
                
                
            

    def predict(self,point):
#         # predcit which cluster each data point belongs to 
 # #         print(np.asarray(self.centroid).shape)
        
#          for x in self.centroid:
#                  # calculation of euclidean distance 
# # #                 # do we include distances ?
#              c.append([np.linalg.norm(data - self.final_centroids[x])])
        
# #         print(len(c))
#          class_idx = np.argmin(np.asarray(c))
                 # once we know which cluster data point belongs to 
#                 # add to the class where each idx has a list
        
         c = [np.linalg.norm(point-self.final_centroids[idx]) for idx in self.centroid]
# #         class_idx = 
         class_idx = c.index(min(c))
         return class_idx
    

### fitting 
         
img_seg = K_Means(k= 8)
img_seg.fit(f)

####
###getting a modfied image 

im_cls=[]
mod_im =[]
idx= 0 
for data in f:
    im_cls =img_seg.predict(data)
    r,g,b = img_seg.final_centroids[im_cls][:3]
    data[:3] = r,g,b
    mod_im.append(data)
    
mod_im = np.asarray(mod_im)

######

## normalization and getting the data 
data = pd.DataFrame(mod_im, columns = ['r', 'g', 'b','i','j'])
#  Inverting the normalization of the image
data['r'] = (data['r'] * np.std(f_space[:,0]) ) + np.mean(f_space[:,0])
data['g'] = (data['g'] * np.std(f_space[:,1]) )+ np.mean(f_space[:,1])
data['b'] = (data['b'] * np.std(f_space[:,2]) )+  np.mean(f_space[:,2])
data['i'] = feature_nn[:,-2]
data['j'] = feature_nn[:,-1]



###########################################################

## getting the  stnadardize

w , h = image.size
x = np.asarray(image)
segmented_image = np.zeros((x.shape))
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        a = data[(data['i'] == i) & (data['j'] == j)]['r_cp']
        b = data[(data['i'] == i) & (data['j'] == j)]['g_cp']
        c = data[(data['i'] == i) & (data['j'] == j)]['b_cp']
        segmented_image[i, j,0] = a
        segmented_image[i, j,1] = b
        segmented_image[i, j,2] = c
        
#############
        # saving the image


im = Image.fromarray(np.uint8(segmented_image))

im.save("segmented-image.jpeg")










