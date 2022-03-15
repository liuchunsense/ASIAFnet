import os
import os.path as osp
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import json
import torchvision.transforms.functional as F
import math

# the Normaliztion of input
normalize = transforms.Normalize(mean=[0.416, 0.42, 0.412],
                                 std=[0.204, 0.2, 0.209])

# 图像的预处理类，包括图像反转，输入图像帧尺寸和预测真值尺寸，归一化，图像反转，颜色变化，随机缩放
# 后来发现图像预处理意义不大，这里只用了归一化

class MyTransforms(object):
    def __init__(self, scale=(0.95, 1.0), ratio=(3. / 4., 4. / 3.), frame_size=(768, 1024),
                 target_size=(384, 512), normalize=None, interpolation=Image.BILINEAR,
                 data_flip=False, data_random=False, color=False):
        if isinstance(frame_size, tuple) and isinstance(target_size, tuple):
            self.frame_size = frame_size
            self.target_size = target_size
        else:
            self.frame_size = (frame_size, frame_size)
            self.target_size = (target_size, target_size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            assert "range should be of kind (min, max)"
        self.color = color
        self.data_flip = data_flip
        self.data_random = data_random
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.normalize = normalize

    @staticmethod
    def get_params(img, scale, ratio):

        width, height = img.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, temp_frame, temp_target, temp_fixation):
        
        # 图像缩放
        temp_frame = [F.resize(frame, self.frame_size, self.interpolation) for frame in temp_frame]              
        temp_target = F.resize(temp_target, self.target_size, self.interpolation)
        temp_fixation = F.resize(temp_fixation, self.target_size, self.interpolation)
           
        # PIL图像文件转tensor
        temp_frame = [transforms.ToTensor()(frame) for frame in temp_frame]                 
        temp_target = transforms.ToTensor()(temp_target)
        temp_fixation = transforms.ToTensor()(temp_fixation)

        if self.normalize is not None:
            temp_frame = [self.normalize(frame) for frame in temp_frame]

        return temp_frame, temp_target, temp_fixation

# 数据加载文件，没有目标显著性估计
class Gaze_not_object(Dataset):
    def __init__(self, data_path, im_or_video=True,txt_dir = None,  datatype=None, data_dir=None):
        super(Gaze_not_object, self).__init__()
        self.data_path = data_path           #
        self.data_dir = data_dir
        
        # 导入图像变换函数
        self.transform = MyTransforms(normalize=normalize, data_flip=True, color=True) \
            if datatype == 'train' else MyTransforms(normalize=normalize, color=False)

        # 判断文件是否存在
        if not osp.exists(self.data_dir):
            raise RuntimeError("json file is not exist")
        if not osp.exists(self.data_path):
            raise RuntimeError("data path is not exist")
        
        self.files = []    #用于数据集的图像名称
        self.datatype = datatype       # 训练集or测试集？
        self.im_or_video = im_or_video     # 这里是一个静态图像与动态图像的标志位， True输出一张图像，False输出一组连续图像
        for line in open(self.data_path):   # 导入json文件，包括图像名称，相对路径
            videos = json.loads(line)

        for video_inf in videos:
            video_class, video_name = video_inf[0]

            video = osp.join(self.data_dir, video_class, video_name)
            frames_path = osp.join(video, 'images')

            frames = os.listdir(frames_path)

            for ind in range(len(frames)):
                if ind == 0:
                    continue
                self.files.append(
                    {
                        "frame_ind": ind,                                     # 第ind帧图像
                        "video_name": video,                                  # 第ind帧图像所对应的video路径， 即根路径
                        "video_info": osp.join(video_class, video_name)       # 第ind帧图像所对应的类别
                    }
                )

        print("{} {} Image are load".format(self.datatype, len(self.files)))

    def __getitem__(self, item):
        file = self.files[item]
        frame_ind, video_name, video_info = file["frame_ind"], file["video_name"], file["video_info"]
        frame_path = osp.join(video_name, 'images/%04d.png' % (frame_ind + 1))
        last_path = osp.join(video_name, 'images/%04d.png' % (frame_ind))

        temp_frame = []
        temp_frame.append(Image.open(frame_path).convert('RGB'))
        if not self.im_or_video: temp_frame.append(Image.open(last_path).convert('RGB'))    #在这里决定输出1帧还是两帧图像

        target_path = osp.join(video_name, 'maps/%04d.png' % (frame_ind + 1))
        fixation_path = osp.join(video_name, 'fixation/%04d.png' % (frame_ind + 1))

        temp_target = Image.open(target_path).convert('L')
        temp_fixation = Image.open(fixation_path).convert('L')

        temp_frame, temp_target, temp_fixation = self.transform(temp_frame, temp_target, temp_fixation)

        temp_frame = torch.stack(temp_frame, dim=0)

        return temp_frame, temp_target, temp_fixation ,video_name , '%04d.png' % (frame_ind + 1)

    def __len__(self):
        return len(self.files)

# 数据加载文件，包含目标显著性估计
class Gaze_object(Dataset):
    def __init__(self, data_path, txt_dir = None, im_or_video=True, datatype=None, data_dir=None):
        super(Gaze_object, self).__init__()
        self.alpha = 0.54
        self.down_ratio = 2
        self.wh_area_process = 'log'

        self.data_path = data_path
        self.data_dir = data_dir
        self.transform = MyTransforms(normalize=normalize, data_flip=True, color=True) \
            if datatype == 'train' else MyTransforms(normalize=normalize, color=False)
        self.datatype = datatype
        self.im_or_video = im_or_video
        self.txt_dir = txt_dir

        if not osp.exists(self.data_dir):
            raise RuntimeError("json file is not exist")
        if not osp.exists(self.data_path):
            raise RuntimeError("data path is not exist")
        self.files = []

        self.video_dict = {}
        self.num_fg = 8
        self.label2num = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4, '7': 5, '9': 6, '11': 7}     #8类目标的转换，从原有的COCO类别代号转变为0-7
        for line in open(data_path):
            videos = json.loads(line)

        for video_inf in videos:
            video_class, video_name = video_inf[0]

            video = osp.join(self.data_dir, video_class, video_name)
            frames_path = osp.join(video, 'images')

            frames = os.listdir(frames_path)

            for ind in range(len(frames)):
                if ind == 0:
                    continue
                self.files.append(
                    {
                        "frame_ind": ind,
                        "video_name": video,
                        "video_info": osp.join(video_class, video_name)
                    }
                )

        print("{} {} Image are load".format(self.datatype, len(self.files)))

    def __len__(self):
        return len(self.files)
    
    ############### 目标检测信息转换程序段##################
    # 把目标的位置和类别信息转换成一个语义特征图
    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        gt_boxes = torch.Tensor(gt_boxes)
        gt_labels = torch.Tensor(gt_labels)

        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = self.bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = self.bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = self.bbox_areas(gt_boxes)

        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k].long()

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item() * 2, w_radiuses_alpha[k].item() * 2)
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

        return heatmap
    ############### 目标检测信息转换程序段##################
    
    # 由于图像中的目标数量不定，这里通过过采样的方式来扩充
    def gt_expand(self, bboxs, labels):
        new_bboxs, new_labels = [], []
        bbox_len = len(bboxs)
        for ind in range(30):
            new_bboxs.append(bboxs[ind % bbox_len])
            new_labels.append(labels[ind % bbox_len])

        new_bboxs = torch.Tensor(new_bboxs)
        new_labels = torch.Tensor(new_labels)
        return new_bboxs, new_labels

    def __getitem__(self, item):
        file = self.files[item]
        frame_ind, video_name, video_info = file["frame_ind"], file["video_name"], file["video_info"]
        frame_path = osp.join(video_name, 'images/%04d.png' % (frame_ind + 1))
        last_frame_name = osp.join(video_name, 'images/%04d.png' % frame_ind)

        temp_frame = []
        temp_frame.append(Image.open(frame_path).convert('RGB'))
        if not self.im_or_video: temp_frame.append(Image.open(last_frame_name).convert('RGB'))
        
        # 读取目标显著性文件真值
        bbox_path = osp.join(self.txt_dir, video_info, 'importance/%04d.txt' % (frame_ind + 1))
        obj_cls = []
        if osp.exists(bbox_path):
            bboxs, labels, clss = [], [], []
            with open(bbox_path, 'r') as lines:
                for line in lines:
                    cls, x1, y1, x2, y2, importance = line.split()     # cls目标类别，【x1, y1, x2, y2】坐标信息 ，importance 目标的显著性类别
                    cls = self.label2num[cls]
                    obj_cls.append(cls)
                    clss.append(cls)
                    bboxs.append([int(x1), int(y1), int(x2), int(y2)])
                    if importance == '0':
                        labels.append([1, 0])                          #由于显著性类别编码
                    else:
                        labels.append([0, 1])
            
            # 热力图生成
            heatmap = self.target_single_image(bboxs, clss,                               
                                               [int(temp_frame[-1].size[1] / self.down_ratio),
                                                int(temp_frame[-1].size[0] / self.down_ratio)])
        
        # 对于没有目标的图像，这里使用[[0, 0, 0, 0]]出的像素代替
        else:
            bboxs = [[0, 0, 0, 0]]
            labels = [[0, 0]]
            heatmap = torch.zeros((self.num_fg, int(temp_frame[-1].size[1] / self.down_ratio),
                                   int(temp_frame[-1].size[0] / self.down_ratio)))

        if self.datatype == 'train':
            bboxs, labels = self.gt_expand(bboxs, labels)
        else:
            bboxs = torch.Tensor(bboxs)

        labels = torch.Tensor(labels)
        obj_cls = torch.Tensor(obj_cls)

        target_path = osp.join(video_name, 'maps/%04d.png' % (frame_ind + 1))                        
        fixation_path = osp.join(video_name, 'fixation/%04d.png' % (frame_ind + 1))

        temp_target = Image.open(target_path).convert('L')
        temp_fixation = Image.open(fixation_path).convert('L')

        temp_frame, temp_target, temp_fixation = self.transform(temp_frame, temp_target, temp_fixation)

        temp_frame = torch.stack(temp_frame, dim=0)
        
        # 这里的输出需要对文件类别进行判断，因为我们测试集用的batchsize是1，所以可以输出文字信息，但是训练集不可以。
        if self.datatype == 'train':
            return temp_frame, temp_target, temp_fixation, heatmap, bboxs, labels
        else:
            return temp_frame, temp_target, temp_fixation, heatmap, bboxs, labels, video_name , '%04d.png' % (frame_ind + 1), obj_cls









