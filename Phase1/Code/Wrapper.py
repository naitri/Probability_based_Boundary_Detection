#!/usr/bin/env python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Naitri Rajyaguru (nrajyagu@umd.edu)
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import skimage.transform
import sklearn.cluster
import argparse
import math
def plot_save(filters,file_dir,cols):
    rows = math.ceil(len(filters)/cols)
    plt.subplots(rows, cols, figsize=(15,15))
    for index in range(len(filters)):
        plt.subplot(rows, cols, index+1)
        plt.axis('off')
        plt.imshow(filters[index], cmap='gray')
    plt.savefig(file_dir)
    plt.close()

def gaussian2d(scale, size):
    size = size/2
    sigma = scale
    x,y = np.ogrid[-size:size+1,-size:size+1]
    normal = 1 / np.sqrt(2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def DoGfilter(scale=[3,5], orientation = 16, size = 49):
    filter_bank = []
    Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
    orient = np.linspace(0,360,orientation)
    for s in scale:
        
        G = gaussian2d(s, size)
        Gx = cv2.filter2D(G,-1, Sx)
        Gy = cv2.filter2D(G,-1, Sy)
        for i in enumerate(orient):
            filters = (Gx * np.cos(i[1])) +  (Gy * np.sin(i[1]))
            filter_bank.append(filters)
    
    return filter_bank

def gaussian2d_variation(sigma,size):
	size = size/2
	sigmax = sigma[0]
	sigmay = sigma[1]
	x,y = np.ogrid[-size:size+1,-size:size+1]
	x_term = x**2 / (2* sigmax**2)
	y_term = y**2 / (2* sigmay**2)
	gu =  np.exp(-(x_term + y_term))
	return gu


def LMFilter(scales, orientations, size, type_filter):
	    l_scales = scales
	    if type_filter == "LML":
	        g_scales = scales
	    else:
	        g_scales = scales[0:3]
	    gauss1D = []
	    gauss2D = []
	    gaussian = []
	    LoG =[]
	    orients=np.linspace(0,360,orientations)
	    Kx = -1*np.array([[-1,0,1]])
	    Ky = -1*np.array([[-1],[0],[1]])
	   
	    # 4 Gaussian Filters
	    for scale in scales:
	        sigma = [scale, scale]
	        gauss = gaussian2d_variation(sigma, size)
	        gaussian.append(gauss)
	#         plt.imshow(gauss,cmap='binary')
	#         plt.show()

	    
	    # 1st & 2nd derivatives
	   
	    for s in g_scales:
	        sigma = [s, 3*s]
	        G = gaussian2d_variation(sigma,size)
	      
	        first_deri_gaussian = cv2.filter2D(G, -1, Kx) + cv2.filter2D(G, -1, Ky)
	        second_deri_gaussian = cv2.filter2D(first_deri_gaussian, -1, Kx) + cv2.filter2D(first_deri_gaussian, -1, Ky)
	      
	        for i in enumerate(orients):
	            gauss_1D = skimage.transform.rotate(first_deri_gaussian,i[1])
	            gauss_2D = skimage.transform.rotate(second_deri_gaussian,i[1])
	            gauss1D.append(gauss_1D)
	            gauss2D.append(gauss_2D)
	            
	            
	        #Laplacian of Gaussian filters
	    
	    for scale in l_scales:
	        s = [scale, scale]
	        G = gaussian2d_variation(s,size)
	        kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
	        log= cv2.filter2D(G, -1,kernal)
	#         plt.imshow(log,cmap = 'binary')
	#         plt.show()
	        LoG.append(log)



	    for scale in l_scales:
	        s = [3*scale, 3*scale]
	        G = gaussian2d_variation(s,size)
	        kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
	        log= cv2.filter2D(G, -1,kernal)
	#         plt.imshow(log,cmap = 'binary')
	#         plt.show()
	        
	        LoG.append(log)
	           
	    
	   
	    return gauss1D + gauss2D + LoG + gaussian


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def GaborFilterBank(sigmas, theta, Lambda, psi, gamma,num_filters):
    gabor_bank = []
    angle = np.linspace(0,360,num_filters)
    #for different sigma
    for sigma in sigmas:
#         print(sigma)
        gabor_ = gabor(sigma, theta=0.25, Lambda=1, psi=0.5, gamma=1)
        for i in enumerate(angle):
            gabor_rot = skimage.transform.rotate(gabor_,i[1])
            gabor_bank.append(gabor_rot)
#             plt.imshow(gabor_rot,cmap = 'binary')
#             plt.show()
            

            
    return gabor_bank

def HalfDisk(radius,angle):
    size = 2*radius + 1
    mask = np.ones([size,size])
    for i in range(radius):
        for j in range(size):
            dist = (i-radius)**2 + (j-radius)**2
            if (dist < radius**2):
                mask[i,j] = 0
    mask = skimage.transform.rotate(mask,angle,cval=1)
    mask = np.round(mask)
#     mask = scipy.ndimage.rotate(mask,angle)

    return mask

def HalfDiskFilterBank(radius, orientations):
    filter_bank = []
    orients=np.linspace(0,360,orientations)
    for radii in radius:
        for orient in orients:
            half_mask = HalfDisk(radii,orient)
            half_mask_rot = skimage.transform.rotate(half_mask,180,cval=1)
            half_mask_rot = np.round(half_mask_rot)
            filter_bank.append(half_mask)
            filter_bank.append(half_mask_rot)
#             plt.imshow(half_mask,cmap='gray')
#             plt.show()
   
    return filter_bank
            
def TextonMap(img, DoG, LM, GB):
    maps = np.array(img)
    for i in range(len(DoG)):
        conv = cv2.filter2D(img,-1, DoG[i])
        maps = np.dstack((maps,conv))
        
    for i in range(len(LM)):
        conv = cv2.filter2D(img,-1, LM[i])
        maps = np.dstack((maps,conv))
        
    for i in range(len(GB)):
        conv = cv2.filter2D(img,-1, GB[i])
        maps = np.dstack((maps,conv))
    
    maps = maps[:,:,1:]
    return maps
                
def Texton(img, num):
    p,q,r= img.shape
    img = np.reshape(img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    print(labels.shape)
    l = np.reshape(labels,(p,q))
    # print(l.shape)
    # plt.imshow(l)
    # plt.show()

    return l
def ChiSquareDist(Img, bins,filter1,filter2):
    Img = np.float32(Img)
    chi_sqr_dist = Img.copy()
    tmp = np.zeros(Img.shape)
    for i in range(bins):
        #numpy.ma.masked_where(condition, a, copy=True)[source]
        #Mask an array where a condition is met.
        
        tmp[Img == i] = 1.0
        tmp[Img != i] = 0.0
        g = cv2.filter2D(tmp,-1,filter1)
        h = cv2.filter2D(tmp,-1,filter2)
        chi_sqr_dist = chi_sqr_dist + ((g-h)**2 /(g+h+np.exp(-10)))
    return chi_sqr_dist/2

def gradient(img, bins, half_disk):
    for i in range(int(len(half_disk)/2)):
        left = half_disk[i]
        right = half_disk[i+1]
        dist = ChiSquareDist(img,bins, left,right)
        grad = np.dstack((img,dist))
        grad = grad[:,:,1:]
    return grad

def BrightnessMap(img, num):
    p,q,r= img.shape
    img = np.reshape(img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    l = np.reshape(labels,(p,q))
#     plt.imshow(l,cmap = 'binary')
#     plt.show()
    return l


def ColorMap(img, num):
    p,q,r= img.shape
    img = np.reshape(img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    print(labels.shape)
    l = np.reshape(labels,(p,q))
    print(l.shape)
#     plt.imshow(l)
#     plt.show()
    return l

def main():



	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	dog_filters = DoGfilter(scale=[3,5], orientation = 16, size = 49)
	plot_save(dog_filters,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/DoG.png",8)


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	lm_filters_lms = LMFilter([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49, "LMS")
	plot_save(lm_filters_lms,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/lms.png",12)

	lm_filters_lml = LMFilter([1, np.sqrt(2), 2, 2*np.sqrt(2),4], 6, 49, "LML")
	plot_save(lm_filters_lms,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/lml.png",9)

	lm_filter = lm_filters_lms + lm_filters_lml
	plot_save(lm_filters_lms,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/LM.png",12)


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gabor_filters=GaborFilterBank(sigmas=[5,10,15], theta=0.25, Lambda=1, psi=1, gamma=1,num_filters=16)
	plot_save(gabor_filters,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/Gabor.png",6)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk = HalfDiskFilterBank([3,5,7], 16)
	plot_save(half_disk,"/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/halfdisk.png",10)

	"""
	Read all the images and test pb_lite
	"""
	path = "/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/Images/"
	for i in range(10):
		n = i+1
		img = plt.imread(path+str(n)+'.jpg')
		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		"""
		texton_map = TextonMap(img, dog_filters, lm_filter, gabor_filters)


		"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		texton = Texton(texton_map,64)
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/TextonMap/TextonMap_"+ str(n)+".png", texton)




		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		# """
		texton_gradient = gradient(texton,64,half_disk)
		texton_gradient =texton_gradient[:,:,0]
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/TextonGradient/Tg_"+ str(n)+".png", texton_gradient)

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		brightness_map =BrightnessMap(img,16)
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/BrightnessMap/BrightnessMap_"+ str(n)+".png", brightness_map)

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		brightness_gradient = gradient(brightness_map,16,half_disk)
		brightness_gradient =brightness_gradient[:,:,0]
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/BrightnessGradient/Bg_"+ str(n)+".png", brightness_gradient)

		"""
		Generate Color Map
		Perform color binning or clustering
		# """
		color_map = ColorMap(img,16)
		
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/ColorMap/colormap_"+ str(n)+".png", color_map)

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		color_gradient = gradient(color_map,16,half_disk)
		color_gradient =color_gradient[:,:,0]
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/ColorGradient/Cg_"+ str(n)+".png", color_gradient)

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		sobel = plt.imread('/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/'+str(n)+'.png')


		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		canny = plt.imread('/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/'+str(n)+'.png')

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		temp = (texton_gradient+brightness_gradient+color_gradient)/3.0

		pb_lite = np.multiply(temp, (0.8*canny+0.2*sobel))
		plt.imsave("/home/naitri/Downloads/YourDirectoryID_hw0/Phase1/results/PbLite/PbLite_"+ str(n)+".png", pb_lite)


    
if __name__ == '__main__':
    main()
 


