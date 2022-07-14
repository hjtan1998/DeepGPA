#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Haojiang_Tan
# Datetime： 2021/8/18 19:43 
# IDE： PyCharm


import torch
import torch.optim as optim
import re
from sklearn.decomposition import PCA
import data_generator
import tools
import torch.utils.data as Data
from args import read_args
import copy
from torch.autograd import Variable
import numpy as np
import random


torch.set_num_threads(2)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)



class cross_validation(object):

    def __init__(self, args):

        super(cross_validation, self).__init__()

        self.fold = 5

        self.args = args

        input_data = data_generator.input_data()

        print('Data loading successfully!')

        self.R_X_dict = {
            'R_2_2': input_data.R_2_2,
            'R_4_4': input_data.R_4_4,
            'R_1_2': input_data.R_1_2,
            'R_1_3': input_data.R_1_3,
            'R_1_4': input_data.R_1_4,
            'R_2_4': input_data.R_2_4,

            'X_1_1': input_data.X_1_1,
            'X_1_2': input_data.X_1_2,
            'X_4_1': input_data.X_4_1
        }

        self.gene_num = self.R_X_dict['R_1_4'].shape[0]

        # get layer ID
        lay_id = set()
        for k in self.R_X_dict:
            if (re.split("_", k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]
                if (layer1 == layer2):
                    lay_id.add(layer1)
                else:
                    lay_id.add(layer1)
                    lay_id.add(layer2)

        # Initializing    Gi    GiUt
        G_input_dict = {}
        G_ini_dict = {}
        GiUt_ini_dict = {}
        for i in lay_id:
            for k in self.R_X_dict:
                if(re.split("_", k)[0] == 'R'):
                    layer1 = re.split("_", k)[1]
                    layer2 = re.split("_", k)[2]
                    if (i == layer1):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k]
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k]))
                    if ((i != layer1) & (i == layer2)):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k].T
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k].T))
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    layer_t = re.split("_", k)[2]    # GiUt
                    if (i == layer):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k]
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k]))
                        GiUt_ini_dict['G' + i + 'U' + layer_t] = np.ones((self.R_X_dict[k].shape[1], self.args.ini_d))
            G_ini_dict['G' + i] = self.get_G_ini(G_input_dict['G' + i], self.args.ini_d)


        for i in lay_id:
            for k in self.R_X_dict:
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    layer_t = re.split("_", k)[2]    # GiUt
                    if (i == layer):
                        GiUt_ini_dict['G' + i + 'U' + layer_t] = np.matmul(np.linalg.inv(G_ini_dict['G' + i][:self.args.ini_d]), self.R_X_dict[k][:self.args.ini_d]).T

        self.G_ini_dict = G_ini_dict
        self.GiUt_ini_dict = GiUt_ini_dict


        for k in self.R_X_dict:
            self.R_X_dict[k] = torch.from_numpy(np.array(self.R_X_dict[k])).float()

        for k in self.G_ini_dict:
            self.G_ini_dict[k] = torch.from_numpy(np.array(self.G_ini_dict[k])).float()

        for k in self.GiUt_ini_dict:
            self.GiUt_ini_dict[k] = torch.from_numpy(np.array(self.GiUt_ini_dict[k])).float()


        if torch.cuda.is_available():
            for k in self.R_X_dict:
                self.R_X_dict[k] = self.R_X_dict[k].cuda()
            for k in self.G_ini_dict:
                self.G_ini_dict[k] = self.G_ini_dict[k].cuda()
            for k in self.GiUt_ini_dict:
                self.GiUt_ini_dict[k] = self.GiUt_ini_dict[k].cuda()

        x = torch.linspace(0, self.gene_num-1, self.gene_num)

        torch_dataset = Data.TensorDataset(x)
        self.loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def select_batch(self, dict, type, batch):
        copy_dict = copy.deepcopy(dict)
        if(type == 'R_X'):
            for k in copy_dict:
                if (re.split("_", k)[0] == 'R'):
                    layer1 = re.split("_", k)[1]
                    if (layer1 == '1'):
                        copy_dict[k] = copy_dict[k][batch]
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    if (layer == '1'):
                        copy_dict[k] = copy_dict[k][batch]
        elif(type == 'G_ini'):
            copy_dict['G1'] = copy_dict['G1'][batch]
        return copy_dict


    def get_G_ini(self, G_input, dim):
        pca = PCA(n_components=dim)
        G_ini = pca.fit_transform(G_input)
        return G_ini


    def get_cross_validation(self):

        m = self.R_X_dict['R_1_4'].shape[0]
        n = self.R_X_dict['R_1_4'].shape[1]

        all = m * n
        allIndex = np.zeros((all, 2))

        i = 0
        for col in range(n):
            for row in range(m):
                allIndex[i][1] = col
                allIndex[i][0] = row
                i += 1

        Indices = np.arange(all)
        X = copy.deepcopy(Indices)
        np.random.shuffle(X)

        HMDD = copy.deepcopy(allIndex)
        prediction_score = np.zeros((m, n))

        for cv in range(self.fold):
            print('---------------------------Current fold: %d---------------------------' % (cv+1))
            R14_temp = copy.deepcopy(self.R_X_dict['R_1_4'])
            if cv < self.fold - 1:
                B = HMDD[X[cv * int(all / self.fold):int(all / self.fold) * (cv + 1)]][:]
                for i in range(int(all / self.fold)):
                    R14_temp[int(B[i][0])][int(B[i][1])] = 0
            else:
                B = HMDD[X[cv * int(all / self.fold):all]][:]
                for i in range(all - int(all / self.fold) * (self.fold - 1)):
                    R14_temp[int(B[i][0])][int(B[i][1])] = 0

            R_X_dict = copy.deepcopy(self.R_X_dict)

            R_X_dict['R_1_4'] = R14_temp

            self.model = tools.DMF(args)

            if torch.cuda.is_available():
                self.model.cuda()

            self.model.init_weights()

            self.parameters = self.model.parameters()

            self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)

            for epoch in range(self.args.train_iter_n):

                print('>>fold:: ', cv+1, '   epoch: ', epoch)

                loss_sum = 0

                loss_R14_sum = 0

                for step, batch_x in enumerate(self.loader):  # for each training step

                    loss, loss_R14, Y_1_4 = self.model(self.select_batch(R_X_dict, 'R_X', batch_x[0].numpy()),
                                                       self.select_batch(self.G_ini_dict, 'G_ini', batch_x[0].numpy()),
                                                       self.GiUt_ini_dict, epoch)
                    loss_sum += loss
                    loss_R14_sum += loss_R14

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                print(loss_sum, loss_R14_sum)

            loss, loss_R14, F = self.model(R_X_dict, self.G_ini_dict, self.GiUt_ini_dict, 0)

            test_num = B.shape[0]
            for ii in range(test_num):
                prediction_score[int(B[ii][0])][int(B[ii][1])] = F[int(B[ii][0])][int(B[ii][1])]

            self.prediction_score = prediction_score


    def model_evaluate(self):
        if torch.cuda.is_available():
            res_dict = tools.get_evaluation_metrics(self.prediction_score, self.R_X_dict['R_1_4'].cpu().detach().numpy())
        else:
            res_dict = tools.get_evaluation_metrics(self.prediction_score, self.R_X_dict['R_1_4'].detach().numpy())
        return res_dict



if __name__ == '__main__':
    args = read_args()

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_class<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object = cross_validation(args)

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_train<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.get_cross_validation()


    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_evaluate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    res_dict = model_object.model_evaluate()
    np.save('./result/res_dict.npy', res_dict)




































