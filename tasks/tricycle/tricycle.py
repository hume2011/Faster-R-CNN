import os
import sys
import math
import json
import glob
import random
import numpy as np
import cv2

# 项目根目录
ROOT_DIR = os.path.abspath("../../")

# 导入库
sys.path.append(ROOT_DIR)  # 加入代码库到根目录
from mrcnn.config import Config
from mrcnn import utils


class TMonitorConfig(Config):
    """修改配置文件
    继承并重载Config类
    """
    # 该配置的名字
    NAME = "TMonitor"

    # 在一个GPU上训练，每个GPU训练8张数据
    # Batch size 为 4 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # 类别数量 (包括背景)
    NUM_CLASSES = 1 + 1  # 背景 + 1 三轮车

    # 确定输入尺寸
    IMAGE_MAX_DIM = 1280

    # 设置anchor的尺寸
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)  # anchor的像素边长
    
    # 检测目标的置信度
    DETECTION_MIN_CONFIDENCE = 0.95
    

class TMonitorDataset(utils.Dataset):
    """数据集对象，数据标注格式如下：
    {
     "path": "/home/huyu/dl/project/ZJ-TWB-Detection-End2End/data/all/train/clip1-16.jpg",
     "outputs": {
                 "object": [
                            {
                             "name": "TWB", 
                             "bndbox": {"xmax": 573, "ymax": 495, "ymin": 319, "xmin": 378}
                            }, 
                            {
                             "name": "TWB", 
                             "bndbox": {"xmax": 942, "ymax": 437, "ymin": 254, "xmin": 795}
                            }
                           ]
                }, 
     "time_labeled": 1528094317109, 
     "size": {"width": 1280, "height": 720, "depth": 3}, 
     "labeled": true}
    """

    def load_TMonitor (self, dataset_dir, subset):
        """加载数据信息。
        dataset_dir: 数据集路径
        count: 要生成的图片数
        height, width: 图片的高宽
        """
        # 添加类
        self.add_class("TMonitor", 1, "TWB")

        # 添加图片数据
        anno_dir = os.path.join(dataset_dir, subset, 'annotations')
        json_dirs = glob.glob(os.path.join(anno_dir, '*.json'))
        for json_dir in json_dirs:
            json_data = json.load(open(json_dir))
            path = json_data['path'] #图片路径
            image_id = path.split('/')[-1].split('.')[0] #用图片名作为id
            width, height = json_data['size']['width'], json_data['size']['height']
            objs = json_data['outputs']['object'] #检测目标
            self.add_image("TMonitor", image_id=image_id, path=path,
                           width=width, height=height, objs=objs)

    def load_objs(self, info_id, scale, padding):
        """加载单张图片中的目标类别标签和bbox
        args：
            info_id：图片的id（该id对应image_info中的对象：0,1,2,3...)
            scale：resize的比例
            padding：padding的大小
        return：
            class_ids:目标的类别
            bboxes：类别对应的bounding box
        """
        class_ids = []
        bboxes = []
        info = self.image_info[info_id]
        shape = [info['height'], info['width']]
        objs = info['objs']
        for obj in objs:
            class_name = obj['name']
            for i in self.class_info:
                if i['name'] == class_name:
                    class_id = i['id']
                    class_ids.append(class_id)
            y1,x1,y2,x2 = obj['bndbox']['ymin'],obj['bndbox']['xmin'],\
                          obj['bndbox']['ymax'],obj['bndbox']['xmax']
            bboxes.append(np.array([y1,x1,y2,x2]))
        bboxes = utils.resize_bboxes(bboxes, shape, scale, padding)
        class_ids = np.array(class_ids)
        return class_ids, bboxes
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
