# [2416776, 2378996, 2390058, 2397721, 2380814, 2336771, 2385744, 2337155, 2343497, 2400491, 2370544, 2329481, 2338884, 2357151, 2346773, 2388568, 2383685, 2417802, 2378352, 2405591, 2340698, 2391568, 2360334, 2406223, 2328613, 2369047, 2325661, 2345400, 2356717, 2384303, 2344194, 2414809, 2370916, 2404941, 2338403, 2408825, 2370333, 2413483, 2402944, 2380334, 2412350, 2388989, 2379781, 2351122, 2369516, 2379210, 2407337, 2390322, 2394940, 2340300, 2320630, 2410589, 2388085, 2393186, 2355831, 2388284, 2386421, 2369711, 2359672, 2415989, 2383215, 2368946, 2405946, 2363899, 2405004, 2407937, 2344844, 2387773, 2316003, 2389915, 2417886, 2399366, 2345238, 2335434, 2344903, 2396395, 2320098]
# [2415090, 2346745, 2400944, 2347067, 2344765, 2402595, 2319788, 2350125, 2405111, 2403972, 2386339, 2315674, 2394180, 2379927, 2336852, 2388917, 2416218, 2416204, 2321571, 2357389, 2326141, 2395669, 2333919, 2321051, 2387074, 2339452, 2400248, 2408461, 2397424, 2354293, 2325625, 2390387, 2385366, 2331541, 2357002, 2410274, 2407779, 2325435, 2382669, 2401793, 2394503, 2330142, 2414491, 2360452, 2316472, 2376553, 2401212, 2319311, 2318215, 2354437, 2336131, 2337005, 2335839, 2396732, 2339038, 2329477, 2350190, 2392550, 2343246, 2346562, 2373542, 2393276, 2323189, 2374336, 2392406, 2339625, 2324980, 2414917, 2354396, 2407778, 2357595, 2407333, 2374775, 2379468, 2335991, 2392657, 2408234, 2350687, 2372979, 2354486, 2395307, 2348472, 2335721, 2386269, 2327203, 2361256, 2370307, 2370885, 2328471, 2405254, 2397804, 2323362, 2356194]

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
import argparse
# display plots in this notebook

# Change dir to caffe root or prototxt database paths won't work wrong
import os
print os.getcwd()
os.chdir('..')
print os.getcwd()

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
parser.add_argument('-gpu', dest='gpu_id', help='GPU id(s) to use',
                    default=0, type=int)
parser.add_argument("-start",
                  dest="start_idx", type=int, default=None)
parser.add_argument("-end",
                  dest="end_idx", type=int, default=None)

args = parser.parse_args()




import caffe

data_path = './data/genome/1600-400-20' # 1600 objects, 400 attributes, 20 relations

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())


# Check object extraction
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import cv2

GPU_ID = args.gpu_id   # if we have multiple GPUs, pick one 
caffe.set_device(GPU_ID)  
caffe.set_mode_gpu()
net = None
cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

weights = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'

net = caffe.Net(prototxt, caffe.TEST, weights=weights)

import glob
from tqdm import tqdm

start = args.start_idx
end = args.end_idx
VG_path = '/2t/jackyyeh/dataset/VG'
im_files = glob.glob(VG_path + "/*.jpg")
img_ids = [int(im_file.split('/')[-1].strip('.jpg')) for im_file in im_files]

im_files = im_files[start:end]
# img_ids = img_ids[start:end]
# im_file = '/2t/jackyyeh/im2p/data/genome/im2p_test/2411999.jpg'
# im_files = ['/2t/jackyyeh/im2p/data/genome/im2p_test/%d.jpg' % _id for _id in img_ids]

###########################
# Similar to get_detections_from_im
conf_thresh=0.3
# attr_thresh = 0.08
min_boxes=10
max_boxes=100

n_attrs = 3

num_rois = []
rpn_rois = []
box_scores = []
total_boxes = []
labels = []
labels_conf = []
attrs = []
attrs_conf = []
img_ids = []

bad_ids = []

pointer = 0

for im_idx, im_file in enumerate(tqdm(im_files)):

        

    try:

        im = cv2.imread(im_file)

        scores, boxes, attr_scores, rel_scores = im_detect(net, im)

        img_id = int(im_file.split('/')[-1].strip('.jpg'))
        img_ids.append( img_id )

        # Keep the original boxes, don't worry about the regression bbox outputs
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        blobs, im_scales = _get_blobs(im, None)
        
        cls_boxes = rois[:, 1:5] / im_scales[0]
        cls_prob = net.blobs['cls_prob'].data
        attr_prob = net.blobs['attr_prob'].data
        pool5 = net.blobs['pool5_flat'].data

        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1,cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]
        ############################

        boxes = cls_boxes[keep_boxes]
        objects = np.argmax(cls_prob[keep_boxes][:,1:], axis=1)
        label_conf = np.max(cls_prob[keep_boxes][:,1:], axis=1)

        
    #     attr = np.argmax(attr_prob[keep_boxes][:,1:], axis=1)
    #     attr_conf = np.max(attr_prob[keep_boxes][:,1:], axis=1)
        
        #     get top n_attrs of attributes    
        attr = np.argsort(attr_prob[keep_boxes][:,1:], axis=1)[:, -n_attrs:]
        attr_conf = np.sort(attr_prob[keep_boxes][:,1:], axis=1)[:, -n_attrs:]
            
        # rois[:, 1:5] = rois[:, 1:5] / im_scales[0]

        num_rois.append( len(rois[keep_boxes]) )
        total_boxes.append( boxes )
        # rpn_rois.append( rois[keep_boxes] )
        box_scores.append( max_conf[keep_boxes] )
        labels.append( objects )
        labels_conf.append( label_conf )
        attrs.append( attr )
        attrs_conf.append( attr_conf )
        
    except: 
        img_id = int(im_file.split('/')[-1].strip('.jpg'))
        bad_ids.append( img_id )

    # if (im_idx+1) % 1000 == 0:
    #     print im_idx+1
    

## output file

import pickle

if start == None: start = 0
if end == None: end = len(glob.glob(VG_path + "/*.jpg"))

fn = './VG_bbox_label_attr_%dto%d.pkl' % (start, end)
data = {
    'img_ids': img_ids,
    'num_rois': num_rois,
    'boxes': total_boxes,
    'box_scores': box_scores,
    'labels': labels,
    'labels_conf': labels_conf,
    'attrs': attrs,
    'attrs_conf': attrs_conf,
}

with open(fn, 'w') as f:
    pickle.dump(data, f)

print ('bad ids:')
print (bad_ids)







