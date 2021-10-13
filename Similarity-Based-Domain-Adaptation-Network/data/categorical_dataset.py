import os
from .image_folder import make_dataset_with_labels, make_dataset_classwise
from PIL import Image
from torch.utils.data import Dataset
import random
from math import ceil
import torch

class CategoricalDataset(Dataset):
    def __init__(self):
        super(CategoricalDataset, self).__init__()

    def initialize(self, root, classnames, class_set, 
                  batch_size, seed=None, transform=None, 
                  **kwargs):

        self.root = root
        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_paths[self.root] = {}
        cid = 0
        #
        #来自各域的图片路径按类别存在data_paths[self.root]中
        #
        for c in self.class_set:
            self.data_paths[self.root][cid] = make_dataset_classwise(self.root, c)
            cid += 1

        self.seed = seed
        self.classnames = classnames

        self.batch_sizes = {}
        self.batch_sizes[self.root] = {}
        cid = 0
        for c in self.class_set:
            batch_size = batch_size
            self.batch_sizes[self.root][cid] = min(batch_size, len(self.data_paths[self.root][cid]))
            cid += 1

    def __getitem__(self, index):
        data = {}
        root = self.root
        cur_paths = self.data_paths[root]
        
        if self.seed is not None:
            random.seed(self.seed)

        inds = random.sample(range(len(cur_paths[index])), \
                             self.batch_sizes[root][index])

        path = [cur_paths[index][ind] for ind in inds]
        data['Path'] = path
        assert(len(path) > 0)
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)

            if 'Img' not in data:
                data['Img'] = [img]
            else:
                data['Img'] += [img]

        data['Label'] = [self.classnames.index(self.class_set[index])] * len(data['Img'])
        data['Img'] = torch.stack(data['Img'], dim=0)
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalDataset'

#
#源域和目标域数据按类别存在data_paths['source']和data_paths['target']中
#
class CategoricalSTDataset(Dataset):
    def __init__(self):
        super(CategoricalSTDataset, self).__init__()

    def initialize(self, source_root, target_paths,target_weight,source_weight,
                  classnames, class_set, 
                  source_batch_size, 
                  target_batch_size, seed=None, 
                  transform=None, **kwargs):

        self.source_root = source_root
        self.target_paths = target_paths
        self.target_weight = target_weight
        self.source_weight = source_weight

        self.transform = transform
        self.class_set = class_set
        
        self.data_paths = {}
        self.data_weights = {}

        #
        #源域图片路径按类别存在data_paths['source']中
        #
        self.data_paths['source'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['source'][cid] = make_dataset_classwise(self.source_root, c)
            cid += 1

        #
        #目标域图片路径按类别存在data_paths['target']中
        #
        self.data_paths['target'] = {}
        cid = 0
        for c in self.class_set:
            self.data_paths['target'][cid] = self.target_paths[c]
            cid += 1

        #
        #源域图片权重按类别存在data_weights[source]中
        #
        self.data_weights['source'] = {}
        cid = 0
        for c in self.class_set:
            self.data_weights['source'][cid] = self.source_weight[c]
            cid += 1
        # print("SOURCE_CATEGORY_WEIGHT")
        # print(self.data_weights['source'])


        #
        #目标域图片权重按类别存在data_weights[target]中
        #
        self.data_weights['target'] = {}
        cid = 0
        for c in self.class_set:
            self.data_weights['target'][cid] = self.target_weight[c]
            cid += 1

        # print("TARGET_CATEGORY_WEIGHT")
        # print(self.data_weights['target'])



        self.seed = seed
        self.classnames = classnames

        self.batch_sizes = {}
        for d in ['source', 'target']:
            self.batch_sizes[d] = {}
            cid = 0
            for c in self.class_set:
                batch_size = source_batch_size if d == 'source' else target_batch_size
                self.batch_sizes[d][cid] = min(batch_size, len(self.data_paths[d][cid]))
                cid += 1


    def __getitem__(self, index):
        data = {}
        for d in ['source', 'target']:
            cur_paths = self.data_paths[d]

            # print("datapath")
            # print(len(cur_paths[index]))

            cur_weights = self.data_weights[d]

            # print("curweight")
            # print(len(cur_weights[index]))

            if self.seed is not None:
                random.seed(self.seed)

            inds = random.sample(range(len(cur_paths[index])), \
                                 self.batch_sizes[d][index])

            path = [cur_paths[index][ind] for ind in inds]
            data['Path_'+d] = path

            weight = [cur_weights[index][ind] for ind in inds]
            data['Weight_'+d] = weight

            assert(len(path) > 0)
            for p in path:
                img = Image.open(p).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)

                if 'Img_'+d not in data:
                    data['Img_'+d] = [img]
                else:
                    data['Img_'+d] += [img]

            data['Label_'+d] = [self.classnames.index(self.class_set[index])] * len(data['Img_'+d])

            data['Img_'+d] = torch.stack(data['Img_'+d], dim=0)

        # print("getttemdsds")
        # print(data['Label_target'])
        # print(data['Label_source'])

        # print(data)

        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalSTDataset'

