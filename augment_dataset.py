"""augment_dataset.py: data augmentation for object detection

Data augmentation
T ID : Transformation 
- 1 : Tone mapping # need HDR images as input # done in singlwLDR2HDR dir
- 2 : Contrast and Color corrections OR gamma correction
- 3 : Horizontal Flip
- 4 : Random Crops OR scaled with aspect ratio preserved
- 5 : Scaled with aspect ratio not preserved

Uses Kitti format
"""

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

gamma = 0.4
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
	lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

def read_ann(src_ann):
	# Read annotation and convert to format required by data_aug
	boxes = np.loadtxt(src_ann, ndmin=2) 
	perm = [1,2,3,4,0] 
	return boxes[:,perm]

def save_ann(dest_ann, bboxes):
	perm = [4,0,1,2,3]
	#todo float to int
	np.savetxt(dest_ann, bboxes[:, perm], fmt='%d')

def getTransformation(mode_id):
	# mode id
	if mode_id == 3:
		return RandomHorizontalFlip(1) # flip with probability 100%
	elif mode_id == 4:
		return RandomScale(0.4, diff = False) # (scale (-0.2 to 0.2) aspect ratio maintained
	elif mode_id == 5:
		return RandomScale(0.5, diff = True) # scale (-0.2 to 0.2) aspect ratio not maintained
	else:
		print "Invalid transformation"
		sys.exit(0)
	## Some other transformation
	#transformation = RandomRotate(10)
	#transformation = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])

def transform_image(mode_id, src_im, src_ann, dest_im, dest_ann):
	bboxes = read_ann(src_ann)
	img = cv2.imread(src_im)
	if mode_id == 1:
		# https://www.learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
		img = cv2.imread(src_im, cv2.IMREAD_ANYDEPTH)
		# Tonemap using Durand's method obtain 24-bit color image
		tonemapDurand = cv2.createTonemapDurand(2.2)
		ldrDurand = tonemapDurand.process(img)
		img = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
	elif mode_id == 2:
		# Brightness and contrast alpha*F(i,j) + beta
		# alpha 1  beta 0      --> no change  
		# 0 < alpha < 1        --> lower contrast  
		# alpha > 1            --> higher contrast  
		# -127 < beta < +127   --> good range for brightness values
		#img = cv2.convertScaleAbs(img, alpha=1.6, beta=10)
		
		# gamma correction
		img = cv2.LUT(img, lookUpTable)
	else:	# data_sug lib fn
		transformation = getTransformation(mode_id)
		img, bboxes = transformation(img, bboxes)
	# save image and annotations
	cv2.imwrite(dest_im, img)
	save_ann(dest_ann, bboxes)

def run(src_im_dir, src_ann_dir, dest_im_dir, dest_ann_dir, im_ext, ann_ext):
	for file_name in os.listdir(src_im_dir):
		if not file_name.endswith(im_ext):
			continue
		fileno = file_name.split('.')[0]
		src_im = os.path.join(src_im_dir, file_name)
		src_ann = os.path.join(src_ann_dir, fileno+ann_ext)
		for mode_id in [2,3,4,5]:
			mode = "_t{}".format(mode_id)
			dest_im = os.path.join(dest_im_dir, fileno+mode+im_ext[0])
			dest_ann = os.path.join(dest_ann_dir, fileno+mode+ann_ext)
			transform_image(mode_id, src_im, src_ann, dest_im, dest_ann)
			if not os.path.isfile(dest_im):
				print "Transformation not done {}".format(dest_im)

if __name__ == '__main__':
	#data_home = "/home/chrystle/unique_mavi_v2"
	data_home = "/home/chrystle/IDD"
	src_im_dir = os.path.join(data_home, "img")
	src_ann_dir = os.path.join(data_home, "annotation")
	dest_im_dir = os.path.join(data_home, "augmented_img")
	dest_ann_dir = os.path.join(data_home, "augmented_annotation")
	im_ext = (".jpg", ".png") # Src image file list
	ann_ext = ".txt" # Src annotation extention
	run(src_im_dir, src_ann_dir, dest_im_dir, dest_ann_dir, im_ext, ann_ext)

