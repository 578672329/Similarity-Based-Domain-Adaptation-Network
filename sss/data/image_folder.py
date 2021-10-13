###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import collections

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
#
#文件的图片格式是IMG_EXTENSIONS的一种即返回true
#

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#
#dir：数据集的路径
#classnames：类别名的集合
#只对classnames里面的类别的数据的标签和图片路径返回
#

def make_dataset_with_labels(dir, classnames):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels

#
#根据类别获取某类别所有图片的路径
#

def make_dataset_classwise(dir, category):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname != category:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
