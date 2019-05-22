"""kitti2pascal.py: convert annotation from kitti to pascal voc format
Reference: https://github.com/umautobots/vod-converter/blob/master/vod_converter/voc.py
"""

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def read_ann(src_ann):
	return np.loadtxt(src_ann, ndmin=2) 

def add_sub_node(node, name, kvs):
	subnode = ET.SubElement(node, name)
	for k, v in kvs.items():
		add_text_node(subnode, k, v)
	return subnode

def add_text_node(node, name, text):
	subnode = ET.SubElement(node, name)
	subnode.text = str(text)
	return subnode

def getLabel(cid):
	# Mavi label
	label_map = {20:"cow", 17:"dog"}
	return label_map[cid]

def kitti2pascal(src_im, src_ann, dest_ann):
	dt = read_ann(src_ann)
	filename = os.path.basename(src_im)
	folder = "augmented_img"

	xml_root = ET.Element('annotation')
	add_text_node(xml_root, 'filename', filename)
	add_text_node(xml_root, 'folder', folder)
	add_text_node(xml_root, 'segmented', 0)
	im = cv2.imread(src_im)
	width, height, _ = im.shape
	## todo read image and get dims
	add_sub_node(xml_root, 'size', {
			'depth': 3,
			'width': width,
			'height': height
	})
	add_sub_node(xml_root, 'source', {
			'annotation': 'Dummy',
			'database': 'Unknown',
			'image': 'Dummy'
	})

	for row in dt:
		x_object = add_sub_node(xml_root, 'object', {
			'name': getLabel(row[0]),
			'difficult': 0,
			'occluded': 0,
			'truncated': 0,
			'pose': 'Unspecified'
		})
		add_sub_node(x_object, 'bndbox', {
			'xmin': row[1],
			'xmax': row[2],
			'ymin': row[3],
			'ymax': row[4]
		})
	ET.ElementTree(xml_root).write(dest_ann)


def run(src_im_dir, src_ann_dir, dest_ann_dir):
	for file_name in os.listdir(src_im_dir):
		if not file_name.endswith(('.jpg', '.png')): #TODO
			continue
		fileno = file_name.split('.')[0]
		src_im = os.path.join(src_im_dir, file_name)
		src_ann = os.path.join(src_ann_dir, fileno+".txt")
		dest_ann = os.path.join(dest_ann_dir, fileno+".xml")
		kitti2pascal(src_im, src_ann, dest_ann)

if __name__ == '__main__':
	data_home = "/home/chrystle/unique_mavi_v2"
	src_im_dir = os.path.join(data_home, "augmented_img")
	src_ann_dir = os.path.join(data_home, "augmented_annotation")
	dest_ann_dir = os.path.join(data_home, "augmented_annotation")
	run(src_im_dir, src_ann_dir, dest_ann_dir)

