#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Haojiang_Tan
# Datetime： 2021/8/16 20:44 
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
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)


class model_class(object):

    def __init__(self, args):

        super(model_class, self).__init__()

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
                if (re.split("_", k)[0] == 'R'):
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
                    layer_t = re.split("_", k)[2]  # GiUt
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
                    layer_t = re.split("_", k)[2]  # GiUt
                    if (i == layer):
                        GiUt_ini_dict['G' + i + 'U' + layer_t] = np.matmul(
                            np.linalg.inv(G_ini_dict['G' + i][:self.args.ini_d]), self.R_X_dict[k][:self.args.ini_d]).T

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


        self.model = tools.DMF(args)

        if torch.cuda.is_available():
            self.model.cuda()

        self.parameters = self.model.parameters()

        self.model.init_weights()

        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)

        x = torch.linspace(0, self.gene_num - 1, self.gene_num)

        torch_dataset = Data.TensorDataset(x)
        self.loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def select_batch(self, dict, type, batch):
        copy_dict = copy.deepcopy(dict)
        if (type == 'R_X'):
            for k in copy_dict:
                if (re.split("_", k)[0] == 'R'):
                    layer1 = re.split("_", k)[1]
                    if (layer1 == '1'):
                        copy_dict[k] = copy_dict[k][batch]
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    if (layer == '1'):
                        copy_dict[k] = copy_dict[k][batch]
        elif (type == 'G_ini'):
            copy_dict['G1'] = copy_dict['G1'][batch]
        return copy_dict

    def get_G_ini(self, G_input, dim):
        pca = PCA(n_components=dim)
        G_ini = pca.fit_transform(G_input)
        return G_ini


    def model_train(self):

        for epoch in range(self.args.train_iter_n):

            print('>>epoch: ', epoch)

            loss_sum = 0

            loss_R14_sum = 0

            for step, batch_x in enumerate(self.loader):  # for each training step

                loss, loss_R14, Y_1_4 = self.model(self.select_batch(self.R_X_dict, 'R_X', batch_x[0].numpy()),self.select_batch(self.G_ini_dict, 'G_ini', batch_x[0].numpy()),self.GiUt_ini_dict, epoch)

                loss_sum += loss
                loss_R14_sum += loss_R14

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            print(loss_sum, loss_R14_sum)






    def model_save(self):
        print('Model performance trained with all samples.(Note: The indicators obtained here cannot be used to evaluate the predictive performance of the model!!!)')
        loss, loss_R14, Y_1_4 = self.model(self.R_X_dict,self.G_ini_dict,self.GiUt_ini_dict, 0)
        if torch.cuda.is_available():
            tools.get_evaluation_metrics(Y_1_4.cpu().detach().numpy(), self.R_X_dict['R_1_4'].cpu().detach().numpy())
        else:
            tools.get_evaluation_metrics(Y_1_4.detach().numpy(), self.R_X_dict['R_1_4'].detach().numpy())
        torch.save(self.model, './result/DeepGPA_model.pkl')
        print('Model saving is finished!!!')



if __name__ == '__main__':
    args = read_args()

    # model
    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_class<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object = model_class(args)

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_train<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.model_train()

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_save<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.model_save()









