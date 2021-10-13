from torch import nn
from utils.utils import to_cuda
import torch
import numpy as np
# from discrepancy.cddnoweight import CDDNOWEIGHT
import torch.nn.functional as F

class CDD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul, 
                 num_classes, intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes==1)
        self.num_layers = num_layers
        # self.cddnoweight = CDDNOWEIGHT(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
        #           num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
        #           intra_only=self.opt.CDD.INTRA_ONLY)
    
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def split_weight_classwise(self, weight, nums):
        num_classes = len(nums)
        start = end = 0
        weight_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            weight_c = weight[start:end, start:end]
            weight_list += [weight_c]
        return weight_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
	    	2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = to_cuda(torch.zeros_like(dist['st'], requires_grad=False))
        gammas['ss'] = [] 
        gammas['tt'] = [] 
        for c in range(num_classes):
            gammas['ss'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]
            gammas['tt'] += [to_cuda(torch.zeros([num_classes], requires_grad=False))]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end 
                target_end = target_start + nums_T[nt] 
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start, 
                       nums_S[ns]).narrow(1, target_start, nums_T[nt]) 

                gamma = self.gamma_estimation(patch)

                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma

                gammas['st'][source_start:source_end, \
                     target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = to_cuda(torch.stack(gamma_list, dim=0))

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps 
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        ###kernel_val = torch.sum( 系数 * torch.exp(-1.0 * dist), dim=0)
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers 
        kernel_dist = None
        #print("lokkkkk : num_layers = ")
        #print(num_layers)
        for i in range(num_layers):
            # print("iiiii")
            # print(i)

            # print("skdjl")
            # print(dist_layers[i][key])

            dist = dist_layers[i][key] if category is None else \
                      dist_layers[i][key][category]

            # print("ddddisy")
            # print(dist)

            gamma = gamma_layers[i][key] if category is None else \
                      gamma_layers[i][key][category]

            # print(gamma)

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist, 
			gamma, cur_kernel_num, cur_kernel_mul)

                # print("kerneldist")
                # print(kernel_dist)

                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma, 
                  cur_kernel_num, cur_kernel_mul)
            # print("kerneldist")
            # print(kernel_dist)



        return kernel_dist

    def target_patch_weight(self, nums_row, nums_col, dist, weight):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        weight_tensor = to_cuda(torch.zeros([num_classes, num_classes]))
        # print("hsjdhafuasfeasd")
        # print(weight_tensor)
        row_start = row_end = 0
        # print("hhhhhhhelolooo")
        # print(dist)

        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]

                # print("weigthr  ")
                # print(weight.narrow(0, row_start,
                #            nums_row[row]).narrow(1, col_start, nums_col[col]))
                # print("disttt    ")
                # print(dist.narrow(0, row_start,
                #            nums_row[row]).narrow(1, col_start, nums_col[col]))

                # print("targetweight")
                # print(weight.narrow(0, row_start,
                #            nums_row[row]).narrow(1, col_start, nums_col[col]))

                # print("targetdist")
                # print(dist.narrow(0, row_start,
                #            nums_row[row]).narrow(1, col_start, nums_col[col]))

                val = (weight.narrow(0, row_start,
                           nums_row[row]).narrow(1, col_start, nums_col[col])) * (dist.narrow(0, row_start,
                           nums_row[row]).narrow(1, col_start, nums_col[col]))

                val = sum(sum(val))

                # print("val")
                # print(val)

                weight_tensor[row, col] = val


        # print("dfgdhfgdhdfsdfsafasdfasdf")
        # print(weight_tensor)

        return weight_tensor

    def source_patch_weight(self, nums_row, dist, weight, c):

        num_classes = len(nums_row)

        weight_tensor = to_cuda(torch.zeros([1, num_classes]))
        # print("hsjdhafuasfeasd")
        # print(weight_tensor)
        row_start = row_end = 0
        # print("hhhhhhhelolooo")
        # print(dist)

        index_s = 0

        # print("weight")
        # print(weight)
        # print(nums_row)


        for k in range(num_classes):
            if k == c:
                break
            index_s = index_s + nums_row[k]

        col_start = col_end = 0
        for col in range(num_classes):
            col_start = col_end
            col_end = col_start + nums_row[col]

            # print("weigthr  ")
            # print(weight.narrow(0, row_start,
            #            nums_row[row]).narrow(1, col_start, nums_col[col]))
            # print("disttt    ")
            # print(dist.narrow(0, row_start,
            #            nums_row[row]).narrow(1, col_start, nums_col[col]))

            # print("targetweight")
            # print(weight.narrow(0, index_s,
            #                      nums_row[c]).narrow(1, col_start, nums_row[col]))
            #
            # print("targetdist")
            # print(dist[col])
            #
            # print("start")
            # print(index_s)
            # print("end")
            # print(index_e)

            ooo = weight.narrow(0, index_s,
                                 nums_row[c]).narrow(1, col_start, nums_row[col])

            ooo = ooo.cpu()
            # print("sss")
            # print(ooo)
            ooo = ooo.view(1, -1)
            # print("ppp")
            # print(ooo)
            ooo = ooo.squeeze(0).cuda()

            val = ooo * (dist[col])

            # print("vallllll")
            # print(val)


            val = sum(val)

            weight_tensor[0, col] = val


        # print("dfgdhfgdhdfsdfsafasdfasdf")
        # print(weight_tensor)

        return weight_tensor
        
    def compute_paired_dist(self, A, B):

        # print("dddddddddddddddddddddddd")
        # print(A.shape)
        # print("s")
        # print(B.shape)


        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        # print("ddddddddddddddddddddddddsssssssssssssssssss")
        # print(bs_A)
        # print("ss")
        # print(bs_T)
        # print ("sss")
        # print(feat_len)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        # print ("ssss")
        # print(A_expand.shape)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)

        # print ("sssss")
        # print(B_expand.shape)
        dist = (((A_expand - B_expand))**2).sum(2)
        # print ("ssssss")
        # print((A_expand - B_expand))


        # print (dist.shape)
        # ss:10*10  tt:30*30 st:10*30
        # print ((A_expand - B_expand).shape)
        return dist



    def compute_paired_weight(self, A, B):


        # feat_len = len(A[0])

        # A = self.expand_list(A)
        # B = self.expand_list(B)
        # print("aaaa")
        # print(A)

        # A = sum(A, [])
        # print("a")
        # print(A)
        # print("ssss")
        # print(A)
        A_tensor = A

        # B = sum(B, [])
        # print("b")
        # print(B)
        # print("hhhhssss")
        # print(B)
        B_tensor = B

        lt_A = len(A)
        lt_B = len(B)


        A_tensor = torch.tensor(A_tensor)
        A_tensor = A_tensor.cuda()
        A_tensor = A_tensor.unsqueeze(1).expand(lt_A, lt_B)

        B_tensor = torch.tensor(B_tensor)
        B_tensor = B_tensor.cuda()
        B_tensor = B_tensor.unsqueeze(0).expand(lt_A, lt_B)


        C = A_tensor * B_tensor

        # print("iamccccccc")
        # print (C)

        return C

    def compute_weight(self, weight, nums):


        weight = sum(weight, [])
        new_weight = []
        num_classes = len(nums)
        start = end = 0
        weight_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            weight_c = weight[start: end]

            weight_c = np.array(weight_c)
            weight_c = torch.from_numpy(weight_c).cuda()
            weight_c = F.softmax(weight_c)

            for i in range(nums[c]):
                weight_list.append(weight_c[i].item())


        return weight_list

    def compute_noweight(self, weight, nums):

        num_classes = len(nums)

        weight_list = []
        for c in range(num_classes):
            for i in range(nums[c]):
                w = 1/nums[c]
                weight_list.append(w)

        return weight_list






    def forward(self, source, target, nums_S, nums_T, sorce_weight, target_weight):
        assert(len(nums_S) == len(nums_T)), \
             "The number of classes for source (%d) and target (%d) should be the same." \
             % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []
        # weight_layers = []

        weight = {}

        sorce_weight = self.compute_weight(sorce_weight, nums_S)
        target_weight = self.compute_weight(target_weight, nums_T)

        weight['ss'] = self.compute_paired_weight(sorce_weight, sorce_weight)
        weight['tt'] = self.compute_paired_weight(target_weight, target_weight)
        weight['st'] = self.compute_paired_weight(sorce_weight, target_weight)

        # weight['ss'] = self.split_weight_classwise(weight['ss'], nums_S)
        # weight['tt'] = self.split_weight_classwise(weight['tt'], nums_T)

        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]


            dist = {}

            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            # weight['ss'] = self.compute_paired_weight(sorce_weight, sorce_weight)

            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            # weight['tt'] = self.compute_paired_weight(target_weight, target_weight)

            dist['st'] = self.compute_paired_dist(cur_source, cur_target)
            # weight['st'] = self.compute_paired_weight(sorce_weight, target_weight)

            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            # weight['ss'] = self.split_weight_classwise(weight['ss'], nums_S)

            dist['tt'] = self.split_classwise(dist['tt'], nums_T)
            # weight['tt'] = self.split_weight_classwise(weight['tt'], nums_T)

            dist_layers += [dist]

            # weight_layers += [weight]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)


        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.target_patch_weight(nums_S, nums_T, kernel_dist_st, weight['st'])

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):

            # print("sdsdsdsdsds")
            # print(self.kernel_layer_aggregation(dist_layers,
            #                  gamma_layers, 'ss', c).view(num_classes, -1))
            #
            # print("sdsds")
            # print(torch.mean(self.kernel_layer_aggregation(dist_layers,
            #                  gamma_layers, 'ss', c).view(num_classes, -1), dim=1))


            a_kernel_dist_ss = self.source_patch_weight(nums_S, self.kernel_layer_aggregation(dist_layers,
                             gamma_layers, 'ss', c).view(num_classes, -1), weight['ss'], c)
            a_kernel_dist_ss = a_kernel_dist_ss.cpu()
            a_kernel_dist_ss = a_kernel_dist_ss.squeeze(0).cuda()
            kernel_dist_ss += [a_kernel_dist_ss]


            b_kernel_dist_tt = self.source_patch_weight(nums_T, self.kernel_layer_aggregation(dist_layers,
                             gamma_layers, 'tt', c).view(num_classes, -1), weight['tt'], c)
            b_kernel_dist_tt = b_kernel_dist_tt.cpu()
            b_kernel_dist_tt = b_kernel_dist_tt.squeeze(0).cuda()
            kernel_dist_tt += [b_kernel_dist_tt]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)

        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        # print("kkkkkk")
        # print(kernel_dist_ss)
        # print(kernel_dist_tt)
        # print(kernel_dist_st)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes

        inter = None
        if not self.intra_only:
            inter_mask = to_cuda((torch.ones([num_classes, num_classes]) \
                    - torch.eye(num_classes)).type(torch.ByteTensor))
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        cdd = intra if inter is None else intra - inter
        return {'cdd': cdd, 'intra': intra, 'inter': inter}
    def forward2(self, domain, nums, domain_weight):

        num_classes = len(nums)

        # compute the dist
        dist_layers = []
        gamma_layers = []

        # weight_layers = []

        weight = {}

        sorce_weight = self.compute_noweight(domain_weight, nums)
        target_weight = self.compute_weight(domain_weight, nums)

        weight['ss'] = self.compute_paired_weight(sorce_weight, sorce_weight)
        weight['tt'] = self.compute_paired_weight(target_weight, target_weight)
        weight['st'] = self.compute_paired_weight(sorce_weight, target_weight)

        # weight['ss'] = self.split_weight_classwise(weight['ss'], nums_S)
        # weight['tt'] = self.split_weight_classwise(weight['tt'], nums_T)

        for i in range(self.num_layers):
            cur_source = domain[i]
            cur_target = domain[i]

            dist = {}

            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            # weight['ss'] = self.compute_paired_weight(sorce_weight, sorce_weight)

            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            # weight['tt'] = self.compute_paired_weight(target_weight, target_weight)

            dist['st'] = self.compute_paired_dist(cur_source, cur_target)
            # weight['st'] = self.compute_paired_weight(sorce_weight, target_weight)

            dist['ss'] = self.split_classwise(dist['ss'], nums)
            # weight['ss'] = self.split_weight_classwise(weight['ss'], nums_S)

            dist['tt'] = self.split_classwise(dist['tt'], nums)
            # weight['tt'] = self.split_weight_classwise(weight['tt'], nums_T)

            dist_layers += [dist]

            # weight_layers += [weight]

            gamma_layers += [self.patch_gamma_estimation(nums, nums, dist)]

        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.target_patch_weight(nums, nums, kernel_dist_st, weight['st'])

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):

            # print("sdsdsdsdsds")
            # print(self.kernel_layer_aggregation(dist_layers,
            #                  gamma_layers, 'ss', c).view(num_classes, -1))
            #
            # print("sdsds")
            # print(torch.mean(self.kernel_layer_aggregation(dist_layers,
            #                  gamma_layers, 'ss', c).view(num_classes, -1), dim=1))


            a_kernel_dist_ss = self.source_patch_weight(nums, self.kernel_layer_aggregation(dist_layers,
                             gamma_layers, 'ss', c).view(num_classes, -1), weight['ss'], c)
            a_kernel_dist_ss = a_kernel_dist_ss.cpu()
            a_kernel_dist_ss = a_kernel_dist_ss.squeeze(0).cuda()
            kernel_dist_ss += [a_kernel_dist_ss]


            b_kernel_dist_tt = self.source_patch_weight(nums, self.kernel_layer_aggregation(dist_layers,
                             gamma_layers, 'tt', c).view(num_classes, -1), weight['tt'], c)
            b_kernel_dist_tt = b_kernel_dist_tt.cpu()
            b_kernel_dist_tt = b_kernel_dist_tt.squeeze(0).cuda()
            kernel_dist_tt += [b_kernel_dist_tt]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)

        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        # print("kkkkkk")
        # print(kernel_dist_ss)
        # print(kernel_dist_tt)
        # print(kernel_dist_st)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes

        inter = None
        if not self.intra_only:
            inter_mask = to_cuda((torch.ones([num_classes, num_classes]) \
                    - torch.eye(num_classes)).type(torch.ByteTensor))
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))

        # cdd = intra if inter is None else intra - inter
        cdd = intra
        return {'cdd': cdd, 'intra': intra, 'inter': inter}



