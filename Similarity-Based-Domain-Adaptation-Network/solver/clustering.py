import torch
from torch.nn import functional as F
from utils.utils import to_cuda, to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil
import numpy as np

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type 

    #
    #get_dist和cos共同返回A和B的距离，这里的距离用cos度量；其中B可以是一个point sets，这时返回的是一个距离的列表？
    #
    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
		pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))

class Clustering(object):
    def __init__(self, eps, feat_key, max_len=1000, dist_type='cos'):
        self.source_samples = {}
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len
        self.target_dist_list = []
        self.source_dist_list = []
        self.dist2sourcecenter = 0
        self.dist2targetcenter = 0

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    #
    #本次计算的centers和上一次的centers差距很小，停止迭代
    #
    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers) 
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels




    #
    #？？？
    #
    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind


    #？？？？
    #收集聚类要用到的样本，包括数据的特征data_feat，路径data_paths，真实标签data_gt
    #
    def collect_samples(self, net, loader):
        data_feat, data_gt, data_paths = [], [], []

        for sample in iter(loader):
            data = sample['Img'].cuda()
            data_paths += sample['Path']
            if 'Label' in sample.keys():
                data_gt += [to_cuda(sample['Label'])]

            output = net.forward(data)
            feature = output[self.feat_key].data 
            data_feat += [feature]


        self.samples['data'] = data_paths
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
                    if len(data_gt)>0 else None


        self.samples['feature'] = torch.cat(data_feat, dim=0)




    def collect_samples2(self, net, loader):
        data_feat, data_gt, data_paths = [], [], []

        for sample in iter(loader):

            data = sample['Img'].cuda()
            data_paths += sample['Path']
            data_gt += [to_cuda(sample['Label'])]

            output = net.forward(data)
            feature = output[self.feat_key].data
            data_feat += [feature]


        self.source_samples['data'] = data_paths
        self.source_samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None

        self.source_samples['feature'] = torch.cat(data_feat, dim=0)





    def feature_clustering(self, net, loader1, loader2):
        centers = None 
        self.stop = False 

        self.collect_samples(net, loader1)
        feature = self.samples['feature']
########
        self.collect_samples2(net, loader2)
        source_feature = self.source_samples['feature']

        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))
        num_samples = feature.size(0)
        #
        #？？？？
        #
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0 
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)    
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len
    
            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor) 
            centers = mask * centers + (1 - mask) * self.init_centers
            
        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        #samples['dist2center']存每一个目标域数据到每一个聚类中心的距离
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]


        #self.centers存目标域每一类的聚类中心
        print("self.centers")
        print(len(list(self.centers)))

        #init_centers存源域每一类的聚类中心
        print("self.init_centers")
        print(len(list(self.init_centers)))


        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        #num_source_samples = len(self.source_samples['feature'])

        #self.target_dist_list = [[]for i in range(self.num_classes)]
        #self.source_dist_list = [[]for i in range(self.num_classes)]

        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, \
                    self.init_centers))

        for i in range(num_samples):
            self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()



##############dist_list 存各个类别的目标域数据到对应类别的源域中心的距离

        #self.dist2sourcecenter = self.Dist.get_dist(self.samples['feature'], self.init_centers, cross=True)

        #self.dist2targetcenter = self.Dist.get_dist(self.source_samples['feature'], self.centers, cross=True)




######target_dist_list保存目标域样本到同类别源域样本类别中心的距离
        #for i in range(self.num_classes):
            #for k in range(num_samples):
                #if self.samples['label'][k] == i:
                    #self.target_dist_list[i].append(self.dist2sourcecenter[k][i].item())
######对target_dist_list每个类别降序排序
        #for i in range(self.num_classes):
            #self.target_dist_list[i].sort(reverse=True)



        #for i in range(self.num_classes):
            #for k in range(num_source_samples):
                #if self.source_samples['gt'][k] == i:
                    #self.source_dist_list[i].append(self.dist2targetcenter[k][i].item())
######对source_dist_list[i]每个类别降序排序

        #for i in range(self.num_classes):
            #self.source_dist_list[i].sort(reverse=True)









################

        #del self.samples['feature']
        #del self.source_samples['feature']