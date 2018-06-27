"""
Faster R-CNN
配置类
"""

import math
import numpy as np


# 基本配置类。
# 不要直接使用该配置，根据所需配置编写子类并重载该类
# 中需要改变的配置项。

class Config(object):
    """基本配置类
       使用时编写子类并重载需要修改的配置项
    """
    # 该配置的名称
    DATA_NAME = None  # 在子类中重载
    TASK_NAME = None  # 在子类中重载

    # 使用GPU的数量，如果使用CPU训练，上设置成1。
    GPU_COUNT = 1

    # 每个GPU的图片数量
    IMAGES_PER_GPU = 2

    # 每个epoch的训练步数。
    # 不需要与训练集的大小相匹配。
    # 每个epoch结束时都会保存Tensorboard更新，因此将其设置为较小的数字意味着要更频繁地更新TensorBoard。
    # 验证集测试数据也会在每个epoch末期进行计算，并且可能需要一段时间，所以不要将其设置得太小以免花费大量时间在验证集数据上。
    STEPS_PER_EPOCH = 1000

    # 在每个epoch结束时运行的验证步骤数。
    # 更大的数字会提高验证统计的准确性，但会减慢训练速度。
    VALIDATION_STEPS = 50

    # 骨干网络架构。
    # 可设置为: resnet50, resnet101。
    BACKBONE = "resnet101"

    # FPN金字塔每层的步长。
    # 以下基于Resnet101。
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 类别数量(包括背景)
    NUM_CLASSES = 1  # 在子类中重载

    # 锚框的像素边长
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # 每个锚点生成锚框的宽高比
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # 锚点的滑动步长。
    # 如果为1，则在骨干特征层中的每个单元格创建锚点。
    # 如果为2，则每隔一个单元格创建锚点，依此类推。
    RPN_ANCHOR_STRIDE = 1

    # 过滤RPN提议框的非极大抑制阈值。
    # 与极大值重叠率大于该值则丢弃。
    RPN_NMS_THRESHOLD = 0.7

    # 每个图像用于RPN训练的锚点数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # 经过非极大值抑制后保留的ROIs数
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    

    # 输入图片的resizing。
    # 一般来说，使用“square”模式进行训练和预测。
    # 在这种模式下，图像被放大，使得小边= IMAGE_MIN_DIM，
    # 但确保缩放不会使长边大于IMAGE_MAX_DIM，
    # 然后用零填充图像使其成为正方形，这样可以将多个图像放入一个批次中。
    # resizing的模式:
    # none:   不需要resize和pad，返回原图。
    # square: 用零进行Resize和pad，获得正方形的图片，尺寸为[max_dim, max_dim]。
    # pad64:  用零填充宽度和高度使它们成为64的倍数。
    #         如果IMAGE_MIN_DIM或IMAGE_MIN_SCALE不为None，则在填充之前会进行缩放。
    #         在此模式下，IMAGE_MAX_DIM将被忽略。
    #         需要64的倍数来确保功能图在FPN金字塔的6个级别上顺利缩放（2 ** 6 = 64）。
    # crop:   在图像去随机裁剪。
    #         首先，根据IMAGE_MIN_DIM和IMAGE_MIN_SCALE缩放图像，
    #         然后以尺寸为IMAGE_MIN_DIM x IMAGE_MIN_DIM进行随机裁剪。
    #         只能用于训练。
    #         此模式下不使用IMAGE_MAX_DIM。
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # 最小缩放比例，在MIN_IMAGE_DIM之后检查，并可以强制进一步放大。
    # 如果设置为2，那么即使MIN_IMAGE_DIM未做相应设置，图像也会被放大以使宽度和高度加倍。
    # 在'square'模式下，它可以被IMAGE_MAX_DIM推翻。
    IMAGE_MIN_SCALE = 0

    # 图像平均值（RGB）
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 每张图像送入分类网络的ROI数量。
    # 可以通过改变非极大值抑制阀值来增加提议框的数量。
    TRAIN_ROIS_PER_IMAGE = 200

    # 用来训练分类网络的前景提议框的比例
    ROI_POSITIVE_RATIO = 0.33

    # ROI池化生成的正方形特征地图的宽度。
    POOL_SIZE = 7

    # 在一幅图像中使用的ground truth实例的最大数量
    MAX_GT_INSTANCES = 100

    # RPN和最终检测的bounding box细化的标准差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # 最终检测的实例的最大数量。
    DETECTION_MAX_INSTANCES = 100

    # 检测结果的置信度。
    # 小于该值的ROIs被过滤。
    DETECTION_MIN_CONFIDENCE = 0.7

    # 最终检测的非极大值抑制阀值
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate 和 momentum（用于优化器）
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # L2正则化的权重衰减
    WEIGHT_DECAY = 0.0001

    # 损失权重，用于更精确的优化。
    # 可用于R-CNN训练设置。
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.
    }

    # 使用RPN的ROIs或外部生成的ROIs进行训练。
    # 在大多数情况下设置为True。
    # 如果想要对代码生成的ROIs进行培训，而不是来自RPN的ROIs，则设置为False。
    # 例如debug分类器网络时无需RPN。
    USE_RPN_ROIS = True

    # 训练或冻结 batch normalization 层。
    #     None: 训练 BN 层，常规模式。
    #     False: 冻结 BN 层，在batch size较小时效果较好。
    #     True: (禁用)， 即使在预测时也会将BN层设置成训练名模式。
    TRAIN_BN = False  # 由于batch size通常较小，默认设置为False

    # Gradient norm clipping，用于优化器。
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """设置需要计算的属性值。"""
        # batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 输入图片的尺寸
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta 的数据长度。
        # 具体定义查看 compose_image_meta()。
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """显示配置项。"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
