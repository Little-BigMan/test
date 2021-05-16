from cfg import Cfg
from models import Yolov4
from tool.utils import *
from tool.torch_utils import *
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import cv2
import os
import copy
import time
import argparse
#import tool.darknet2pytorch import Darknet
def eval_multi (m, imgfile,num): 
    
#m = Yolov4(yolov4conv137weight = weightfile,n_classes = 1,inference = True)
#m.load.weights(weightfile, map_location = 'cpu') 
  images_dir = os.listdir(imgfile)
  images_dir = [x for x in images_dir if x != '.DS_Store']
  images_dir.sort()
  h_stacks = []
  v_stacks = []
  eval_blend = []
  eval_term = 5
  null_img = np.ones((608, 608, 3), dtype=np.uint8)
  use_cuda = False
  
  for index, name in enumerate(images_dir):
    image_path = os.path.join(data_dir, name)
    blend = cv2.imread(image_path)
    cv2.imwrite("check1.png",blend)
    h, w, c = blend.shape
    if w > h:
        pad = int((w - h) / 2)
        if int(w - h) % 2 == 0:
            blend = cv2.copyMakeBorder(blend , pad, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            blend = cv2.copyMakeBorder(blend , pad, pad + 1, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    else:
      pad = int((h - w) / 2)
      print(pad)
      if int(h - w) % 2 == 0:
          blend = cv2.copyMakeBorder(blend , 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
      else:
          blend = cv2.copyMakeBorder(blend , 0, 0, pad, pad+1, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    blend = cv2.resize(blend, (608, 608))
    cv2.imwrite("check2.png",blend)
    blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
    print(blend.shape)
    boxes = do_detect(m, blend, 0.4, 0.6, use_cuda)
    _,_, blend = plot_boxes_cv2(blend, boxes[0], savename='predictions'+str(index)+'.png', class_names='wine')
    print(blend.shape)
    eval_blend.append(blend)

  for blend in eval_blend:
    h_stacks.append(blend)
    if len(h_stacks) == (eval_term ):
        h_stacks = np.hstack(h_stacks)
        v_stacks.append(h_stacks)
        h_stacks = []

  if len(h_stacks) < (eval_term * 2) and len(h_stacks) != 0:

      fill_num = int((eval_term * 2 - len(h_stacks)) / 2)
      for _ in range(fill_num):
          h_stacks.append(null_img)
          h_stacks.append(eval_term)
      h_stacks = np.hstack(h_stacks)
  if len(v_stacks) == 0:
      v_stacks = h_stacks
  else:
      v_stacks = np.vstack(v_stacks).astype(np.uint8)
  os.makedirs('./result/image/', exist_ok=True)
  cv2.imwrite('./result/image/eval' + str(num)+ '.png', v_stacks)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
#args = get_args()
#print('Loading weights from %s... Done!' % (weightfile))
    #if use_cuda:
    #    m.cuda()
    weight = './Yolov4_epoch38.pth'
    m = Yolov4(weight,1,True)
    data_dir = './data/'
    eval_multi(m, data_dir, 38) 
