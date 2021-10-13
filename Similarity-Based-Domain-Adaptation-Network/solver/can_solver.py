import math

import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from .clustering import DIST


class CANSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, dist_type='cos', **kwargs):
        super(CANSolver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        #
        #bn_domain_map是把源域和目标域映射到0，1标签空间上
        #
        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert('categorical' in self.train_data)

        num_layers = len(self.net.module.FC) + 1

        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                  num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
                  intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}
        self.target_dist_list = []
        self.source_dist_list = []
        self.Dist = DIST(dist_type)
        self.target_similar = []
        self.source_similar = []
        self.source_similar_list = []


    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
		len(self.history['ts_center_dist']) < 1 or \
		len(self.history['target_labels']) < 2:
           return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1], 
			target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1


        while True: 
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                #self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                #
                #
                #update是对目标域进行聚类的入口
                self.update_labels()

                self.clustered_target_samples = self.clustering.samples
                ##
                self.clustered_source_samples = self.clustering.source_samples
                init_centers = self.clustering.init_centers
                ##
                self.target_centers = self.clustering.centers
                self.source_centers = self.clustering.init_centers
                center_change = self.clustering.center_change 
                path2label = self.clustering.path2label

                #target_dist_list = self.clustering.target_dist_list
                #source_dist_list = self.clustering.source_dist_list

                # updating the history
                self.register_history('target_centers', self.target_centers,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
	            	self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                              self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break
                
                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)
            self.loop += 1

        print('Training Done!')

    #
    # 求源域中心来初始化目标域中心
    #
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        #
        #source_dataloader存放源域的聚类batch
        #set_bn_domain把各网络模型的domain设置为当前聚类的domain
        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])


        #
        #用源域中心初始化目标域中心
        #
        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        #
        #targrt_dataloader存放目标域的聚类batch
        #set_bn_domain把各网络模型的domain设置为当前聚类的domain
        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader, source_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples
        num_classes = self.opt.DATASET.NUM_CLASSES

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(
		target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        #gengxin juleizhongxin

        self.target_dist_list = [[]for i in range(num_classes)]
        self.source_dist_list = [[]for i in range(num_classes)]

        ##############dist_list 存各个类别的目标域数据到对应类别的源域中心的距离

        self.dist2targetcenter = self.Dist.get_dist(self.clustered_source_samples['feature'], self.target_centers, cross=True)

        self.dist2sourcecenter = self.Dist.get_dist(chosen_samples['feature'], self.source_centers, cross=True)


        num_samples = len(chosen_samples['feature'])
        num_source_samples = len(self.clustered_source_samples['feature'])

        target_dist_list_unlabel = [[]for i in range(num_classes)]
        source_dist_list_unlabel = [[]for i in range(num_classes)]

        ######target_dist_list保存目标域样本到同类别源域样本类别中心的距离
        for i in range(num_classes):
            for k in range(num_samples):
                if chosen_samples['label'][k] == i:
                    self.target_dist_list[i].append([self.dist2sourcecenter[k][i].item(), k])

        #########target_dist_list_unlabel保存目标域样本到同类别源域样本类别中心的距离,只保存距离，不保存下标，用于softmax
        for i in range(num_classes):
            for k in range(num_samples):
                if chosen_samples['label'][k] == i:
                    target_dist_list_unlabel[i].append(math.exp(-self.dist2sourcecenter[k][i].item()))


        for i in range(num_classes):
            # print("before")
            # print(target_dist_list_unlabel[i])
            target_dist_list_unlabel[i] = np.array(target_dist_list_unlabel[i])
            target_dist_list_unlabel[i] = torch.from_numpy(target_dist_list_unlabel[i]).cuda()
            target_dist_list_unlabel[i] = F.softmax(target_dist_list_unlabel[i])
            # print("aftersoftmax")
            # print(target_dist_list_unlabel[i])
            for k in range(len(target_dist_list_unlabel[i])):
                self.target_dist_list[i][k][0] = target_dist_list_unlabel[i][k]

        # print("last")
        # print(self.target_dist_list)




        # list_tonumpy = []
        # target_dist_list_unlabel = torch.tensor(target_dist_list_unlabel)
        # print(target_dist_list_unlabel)
        # print("distclasswise")
        # print(target_dist_list_unlabel)
        # print(self.target_dist_list)

        # for i in range(len(target_dist_list_unlabel)):
        #     target_dist_list_unlabel[i] = np.array(target_dist_list_unlabel[i])
        #     target_dist_list_unlabel[i] = target_dist_list_unlabel[i].astype(float)
        #
        # target_dist_list_unlabel = np.array(target_dist_list_unlabel)
        #
        # target_dist_list_unlabel = torch.from_numpy(target_dist_list_unlabel)
        #
        # target_dist_list_unlabel = target_dist_list_unlabel.cuda()
        #
        #
        # target_dist_list_unlabel = F.softmax(target_dist_list_unlabel, dim = 1)

        # print("distclasswise")
        # print(target_dist_list_unlabel)
        # print(self.target_dist_list)





        # ######对target_dist_list每个类别降序排序
        # for i in range(num_classes):
        #     self.target_dist_list[i].sort(reverse=True, key = lambda x:x[0])

        ######source_dist_list保存目标域样本到同类别源域样本类别中心的距离
        for i in range(num_classes):
            for k in range(num_source_samples):
                if self.clustered_source_samples['gt'][k] == i:
                    self.source_dist_list[i].append([self.dist2targetcenter[k][i].item(), k])


        #########source_dist_list_unlabel保存目标域样本到同类别源域样本类别中心的距离,只保存距离，不保存下标，用于softmax
        for i in range(num_classes):
            for k in range(num_source_samples):
                if self.clustered_source_samples['gt'][k] == i:
                    source_dist_list_unlabel[i].append(math.exp(-self.dist2targetcenter[k][i].item()))


        for i in range(num_classes):
            # print("before")
            # print(target_dist_list_unlabel[i])
            source_dist_list_unlabel[i] = np.array(source_dist_list_unlabel[i])
            source_dist_list_unlabel[i] = torch.from_numpy(source_dist_list_unlabel[i]).cuda()
            source_dist_list_unlabel[i] = F.softmax(source_dist_list_unlabel[i])
            # print("aftersoftmax")
            # print(target_dist_list_unlabel[i])
            for k in range(len(source_dist_list_unlabel[i])):
                self.source_dist_list[i][k][0] = source_dist_list_unlabel[i][k]





        # ######对source_dist_list[i]每个类别降序排序
        # for i in range(num_classes):
        #     self.source_dist_list[i].sort(reverse=True, key = lambda x:x[0])


        #chushihua xiangsidu liebiao de zhi

        self.target_similar =  [2 for i in range(num_samples)]
        self.source_similar =  [2 for i in range(num_source_samples)]

        #計算目標域每個樣本是否是相似樣本
        for i in range(len(self.target_dist_list)):

            m = len(self.target_dist_list[i])

            if m == 0:
                continue
            for j in range(m):

                k = self.target_dist_list[i][j][1]
                self.target_similar[k] = self.target_dist_list[i][j][0].item()
                #####QU QIAN YI BAN WEI XIANGSI YANGBEN
                # if j <= math.ceil(m/2) :
                #     self.target_similar[k] = 1.00000000
                # else:
                #     self.target_similar[k] = 0.00000000


        # print("target_similiae")
        # print(self.target_similar)




#計算源域每個樣本是否是相似樣本
        for i in range(len(self.source_dist_list)):
            m = len(self.source_dist_list[i])

            if m == 0:
                continue
            for j in range(m):
                k = self.source_dist_list[i][j][1]
                self.source_similar[k] = self.source_dist_list[i][j][0].item()
                #####QU QIAN YI BAN WEI XIANGSI YANGBEN
                # if j <= math.ceil(m/2) :
                #     self.source_similar[k] = 1.00000000
                # else:
                #     self.source_similar[k] = 0.00000000

        # print("target_similiae")
        # print(self.target_similar)



        chosen_samples['target_similar'] = self.target_similar
        chosen_samples['source_similar'] = self.source_similar

######计算 source_similar_classwise


        gt = self.clustered_source_samples['gt']
        source_similar = self.source_similar

        for c in range(num_classes):
            mask = (gt == c)
            gt_c = torch.masked_select(gt, mask) if gt is not None else None
            # print("lisy()gt")
            # print(list(gt_c))
            # print(len(list(gt_c)))
            source_similar_c = [source_similar[k] for k in range(mask.size(0)) if mask[k].item() == 1]
            # print("len()list()gt")
            # print(list(source_similar_c))
            # print(len(list(source_similar_c)))
            # samples_c是字典類型：其中data和similar是list，label是tensor，gt是none
            samples_c = {}
            samples_c['gt'] = gt_c
            samples_c['source_similar'] = source_similar_c

            self.source_similar_list.append(samples_c)


        #cur_feature = chosen_samples['feature']
        #cur_label = chosen_samples['label']
        #refs = to_cuda(torch.LongTensor(range(num_classes)).unsqueeze(1))
        #num_features = len(chosen_samples['feature'])

        #for i in range(num_features):
        #     if chosen_samples['label'][i] not in filtered_classes:
        #        break
        #    gt = to_cuda(cur_label[i])
        #    gt = gt.unsqueeze(0).expand(num_classes, -1)
        #    mask = (gt == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        #    feature = cur_feature[i].unsqueeze(0)
            # update centers
            #filterd_centers += torch.sum(feature * mask, dim=1)

        #chosen_samples['center'] = filterd_centers

        #print(chosen_samples['center'])


        print('The number of filtered classes: %d.' % len(filtered_classes))

        return chosen_samples, filtered_classes


##########


    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader

        #target_classwise是一個list，其中每個元素是一個類別的字典，包括了該類別的樣本路徑，該類別的樣本標簽，該類別的mubiao樣本相似度
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)


        source_similar_classwise = 0


        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames

        #############过滤之后的类别进入训练的阶段
        #############
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}

        #an leibie cunchu  mubiaoyu quanzhong
        dataloader.target_weight = {classnames[c]: target_classwise[c]['target_similar'] \
                      for c in filtered_classes}

        #an leibie cunchu yuanyu quanzhong
        dataloader.source_weight = {classnames[c]: self.source_similar_list[c]['source_similar']  \
                      for c in filtered_classes}


        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')
        #yici chuan yige batch yigong liushige yangben ,yuanyu sanshige mubiaoyu sanshige ,yigong shige leibie
        # print("dhsjdhjs")
        # print(list(samples))
        # print(samples['Label_source'])
        # print(samples['Label_target'])


        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]
        ##quanzhong
        source_weight = samples['Weight_source']

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
        ##quanzhong
        target_weight = samples['Weight_target']
        
        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert(self.selected_classes == 
               [labels[0].item() for labels in  samples['Label_target']])
        return source_samples, source_nums, source_weight, target_samples, target_nums, target_weight
            
    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        #参与训练的两部分数据：一部分是源域数据；一部分是源+目标域数据
        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        self.train_data['categorical']['iterator'] = \
                     iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.net.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0
            source_cdd_loss_iter = 0
            target_cdd_loss_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss
         
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, source_weight_cls, \
                       target_samples_cls, target_nums_cls, target_weight_cls = self.CAS()

                # print("sourcesampleclss")
                # print (source_samples_cls)
                # print(source_nums_cls)
                # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                #
                # print (source_weight_cls)
                # [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]

                #source_samples_cls源域各类别的样本，一共十个类别，每个类别1个样本
                #source_nums_cls源域各类别的样本的个数
                #target_samples_cls
                #target_nums_cls

                # 2) forward and compute the loss
                #把batch内十个类别的源域样本连接起来
                source_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in source_samples_cls], dim=0)
                #把batch内十个类别的目标域样本连接起来
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)


                # prepare the features
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target)

                #print("lasttt")
                #print(feats_toalign_S) 二维，第一维存feat，共十个10*1024，第二维存probs，共十个10*31，每一维都是一个tensor
                #print(len(feats_toalign_S))
                #print(feats_toalign_S[0].shape)
                #print("last")
                #print(feats_toalign_T)  二维，第一维存feat，共三十个，第二维存probs，共三十个，每一维都是一个tensor
                #print(len(feats_toalign_T))
                #print(feats_toalign_T[0].shape)

                # print("feats_toalign_S")
                # print (feats_toalign_S)
                # 两个tensor，分别是十个样本在两层的输出特征
                # print ("a")
                # print (len(feats_toalign_S))
                # 两层，每一层对应一个tensor
                # print ("b")
                # print (feats_toalign_S[0])
                # 第一层网络输出tensor
                # print ("c")
                # print (len(feats_toalign_S[0]))
                # 第一层网络输出的特征共10个，分别对应十个样本

                cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T,
                               source_nums_cls, target_nums_cls, source_weight_cls, target_weight_cls)[self.discrepancy_key]

                cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                cdd_loss.backward(retain_graph=True)

                cdd_loss_iter += cdd_loss
                loss += cdd_loss

                ####计算源域域内的带权重的cdd
                source_cdd_loss = self.cdd.forward2(feats_toalign_S, source_nums_cls,
                                                    source_weight_cls)[self.discrepancy_key]

                source_cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                source_cdd_loss.backward(retain_graph=True)

                source_cdd_loss_iter += source_cdd_loss

                loss += source_cdd_loss

                ######计算目标域域内的带权重的cdd
                target_cdd_loss = self.cdd.forward2(feats_toalign_T, target_nums_cls,
                                                    target_weight_cls)[self.discrepancy_key]

                target_cdd_loss *= self.opt.CDD.LOSS_WEIGHT
                target_cdd_loss.backward()

                target_cdd_loss_iter += target_cdd_loss

                loss += target_cdd_loss





            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter,
			'source_cdd_loss':source_cdd_loss_iter, 'target_cdd_loss':target_cdd_loss_iter, 'total_loss': loss}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu, class_acc = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, accu))
                    print(class_acc)

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

