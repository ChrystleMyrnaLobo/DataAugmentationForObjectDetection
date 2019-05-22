"""pascal2kitti.py: convert annotation from pascal voc to kitti format
Reference: https://github.com/umautobots/vod-converter/blob/master/vod_converter/voc.py

# [label] [x1] [y1] [x2] [y2]
  1 0.0 104.98 500.0 370.61
"""

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def getLabelID(c):
	# MAVI label
	label_map = {"cow":2, "dog":1}
	#label_map = {20:"cow", 17:"dog"}
	return label_map[c]

def read_ann(src_ann):
	tree = ET.parse(src_ann)
	root = tree.getroot()
	boxes = []
	for node in root.findall('object'):
		bndbox = node.find('bndbox')
		label = getLabelID( node.find('name').text )
		ymin = float(bndbox.find('ymin').text) 
		xmin = float(bndbox.find('xmin').text)
		xmax = float(bndbox.find('xmax').text)
		ymax = float(bndbox.find('ymax').text) 
		row = [label, xmin, ymin, xmax, ymax]
		boxes.append(row)
	if len(boxes) > 1:
		print src_ann
	return np.array(boxes)

def pascal2kitti(src_im, src_ann, dest_ann):
	dt = read_ann(src_ann)
	np.savetxt(dest_ann, dt, delimiter=" ", fmt='%i')

def run(src_im_dir, src_ann_dir, dest_ann_dir):
	for file_name in os.listdir(src_im_dir):
		if not file_name.endswith(('.jpg', '.png')): #TODO
			continue
		fileno = file_name.split('.')[0]
		src_im = os.path.join(src_im_dir, file_name)
		src_ann = os.path.join(src_ann_dir, fileno+".xml")
		dest_ann = os.path.join(dest_ann_dir, fileno+".txt")
		pascal2kitti(src_im, src_ann, dest_ann)

if __name__ == '__main__':
	#data_home = "/home/chrystle/unique_mavi_v2"
	data_home = "/home/chrystle/IDD"
	src_im_dir = os.path.join(data_home, "img")
	src_ann_dir = os.path.join(data_home, "annotation")
	dest_ann_dir = os.path.join(data_home, "annotation")
	run(src_im_dir, src_ann_dir, dest_ann_dir)

