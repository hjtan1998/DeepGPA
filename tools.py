#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Haojiang_Tan
# Datetime： 2021/8/9 9:06
# IDE： PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import re
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)


class DMF(nn.Module):
    def __init__(self, args):
        super(DMF, self).__init__()

        self.act = nn.LeakyReLU()

        self.Layers = args.Layers
        self.ini_d = args.ini_d
        self.train_iter_n = args.train_iter_n

        self.Layer_1_1_run = nn.Linear(self.ini_d, self.Layers['Layer_1'][0])
        self.Layer_1_2_run = nn.Linear(self.Layers['Layer_1'][0], self.Layers['Layer_1'][1])

        self.Layer_2_1_run = nn.Linear(self.ini_d, self.Layers['Layer_2'][0])
        self.Layer_2_2_run = nn.Linear(self.Layers['Layer_2'][0], self.Layers['Layer_2'][1])

        self.Layer_3_1_run = nn.Linear(self.ini_d, self.Layers['Layer_3'][0])
        self.Layer_3_2_run = nn.Linear(self.Layers['Layer_3'][0], self.Layers['Layer_3'][1])

        self.Layer_4_1_run = nn.Linear(self.ini_d, self.Layers['Layer_4'][0])
        self.Layer_4_2_run = nn.Linear(self.Layers['Layer_4'][0], self.Layers['Layer_4'][1])



        self.Layer_1_U1_1_run = nn.Linear(self.ini_d, self.Layers['Layer_1'][0])
        self.Layer_1_U1_2_run = nn.Linear(self.Layers['Layer_1'][0], self.Layers['Layer_1'][1])
        self.Layer_1_U2_1_run = nn.Linear(self.ini_d, self.Layers['Layer_1'][0])
        self.Layer_1_U2_2_run = nn.Linear(self.Layers['Layer_1'][0], self.Layers['Layer_1'][1])
        self.Layer_4_U1_1_run = nn.Linear(self.ini_d, self.Layers['Layer_4'][0])
        self.Layer_4_U1_2_run = nn.Linear(self.Layers['Layer_4'][0], self.Layers['Layer_4'][1])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def cos_sim(self, a, b):
        feature1 = F.normalize(a)
        feature2 = F.normalize(b)
        distance1 = feature1.mm(feature2.t())
        return distance1

    def forward(self, R_X_dict, G_ini_dict, GiUt_ini_dict, iter_i):

        G_output_dict = {}
        U_output_dict = {}
        Y_dict = {}

        G_output_dict['G1'] = self.Layer_1_1_run(G_ini_dict['G1'])
        G_output_dict['G1'] = self.act(G_output_dict['G1'])
        G_output_dict['G1'] = self.Layer_1_2_run(G_output_dict['G1'])
        G_output_dict['G1'] = self.act(G_output_dict['G1'])

        G_output_dict['G2'] = self.Layer_2_1_run(G_ini_dict['G2'])
        G_output_dict['G2'] = self.act(G_output_dict['G2'])
        G_output_dict['G2'] = self.Layer_2_2_run(G_output_dict['G2'])
        G_output_dict['G2'] = self.act(G_output_dict['G2'])

        G_output_dict['G3'] = self.Layer_3_1_run(G_ini_dict['G3'])
        G_output_dict['G3'] = self.act(G_output_dict['G3'])
        G_output_dict['G3'] = self.Layer_3_2_run(G_output_dict['G3'])
        G_output_dict['G3'] = self.act(G_output_dict['G3'])

        G_output_dict['G4'] = self.Layer_4_1_run(G_ini_dict['G4'])
        G_output_dict['G4'] = self.act(G_output_dict['G4'])
        G_output_dict['G4'] = self.Layer_4_2_run(G_output_dict['G4'])
        G_output_dict['G4'] = self.act(G_output_dict['G4'])


        U_output_dict['G1U1'] = self.Layer_1_U1_1_run(GiUt_ini_dict['G1U1'])
        U_output_dict['G1U1'] = self.act(U_output_dict['G1U1'])
        U_output_dict['G1U1'] = self.Layer_1_U1_2_run(U_output_dict['G1U1'])
        U_output_dict['G1U1'] = self.act(U_output_dict['G1U1'])


        U_output_dict['G1U2'] = self.Layer_1_U2_1_run(GiUt_ini_dict['G1U2'])
        U_output_dict['G1U2'] = self.act(U_output_dict['G1U2'])
        U_output_dict['G1U2'] = self.Layer_1_U2_2_run(U_output_dict['G1U2'])
        U_output_dict['G1U2'] = self.act(U_output_dict['G1U2'])

        U_output_dict['G4U1'] = self.Layer_4_U1_1_run(GiUt_ini_dict['G4U1'])
        U_output_dict['G4U1'] = self.act(U_output_dict['G4U1'])
        U_output_dict['G4U1'] = self.Layer_4_U1_2_run(U_output_dict['G4U1'])
        U_output_dict['G4U1'] = self.act(U_output_dict['G4U1'])


        for k in R_X_dict:
            if (re.split("_", k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]

                Y_dict['Y_' + layer1 + '_' + layer2] = torch.mm(G_output_dict['G' + layer1],
                                                                G_output_dict['G' + layer2].T)

            if (re.split("_", k)[0] == 'X'):
                layer = re.split("_", k)[1]
                layer_t = re.split("_", k)[2]
                Y_dict['X_' + layer + '_' + layer_t] = torch.mm(G_output_dict['G' + layer],
                                                                U_output_dict['G' + layer + 'U' + layer_t].T)

        loss = 0

        for k in R_X_dict:
            if (re.split('_', k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]
                loss += sum(sum((R_X_dict[k] - Y_dict['Y_' + layer1 + '_' + layer2]) ** 2))

            if (re.split('_', k)[0] == 'X'):
                layer = re.split("_", k)[1]
                layer_t = re.split("_", k)[2]
                loss += sum(sum((R_X_dict[k] - Y_dict['X_' + layer + '_' + layer_t]) ** 2))

        loss_R14 = sum(sum((R_X_dict['R_1_4'] - Y_dict['Y_1_4']) ** 2))

        return loss, loss_R14, Y_dict['Y_1_4']


def get_evaluation_metrics(prediction_score, Y):
    IC = np.zeros((159, 1))
    # Y = (np.abs(Y) + Y) / 2
    m = Y.shape[0]
    n = Y.shape[1]
    prediction_score_col = prediction_score.reshape(-1)
    Indices = np.arange(len(prediction_score_col))
    np.random.shuffle(Indices)
    X = Indices
    thresholds_num = 2000
    X = X[:thresholds_num]
    thresholds = prediction_score_col[X]
    thresholds = np.sort(thresholds)[::-1]  # descend

    tpr = np.zeros((1, thresholds_num)).reshape(-1)
    fpr = np.zeros((1, thresholds_num)).reshape(-1)
    precision = np.zeros((1, thresholds_num)).reshape(-1)
    recall = np.zeros((1, thresholds_num)).reshape(-1)

    Fmax = 0
    Smin = 1000000

    for qq in range(thresholds_num):
        prediction = (prediction_score >= thresholds[qq]) + 0
        TP = np.sum(Y * prediction)
        pre_pos = np.sum(prediction)
        FP = pre_pos - TP
        pre_neg = m * n - pre_pos
        TN = np.sum((prediction + Y == 0) + 0)
        FN = pre_neg - TN

        RU = (prediction - Y == 1) + 0
        ru = np.sum(RU @ IC) / m

        MI = (Y - prediction == 1) + 0
        mi = np.sum(MI @ IC) / m

        s = np.sqrt(ru ** 2 + mi ** 2)
        if (s < Smin):
            Smin = s

        if (TP + FN != 0):
            tpr[qq] = TP / (TP + FN)
            recall[qq] = TP / (TP + FN)
        if (TN + FP != 0):
            fpr[qq] = FP / (TN + FP)
        if (TP + FP != 0):
            precision[qq] = TP / (TP + FP)

        f = 2 * precision[qq] * recall[qq] / (precision[qq] + recall[qq] + 1e-20)
        if (f > Fmax):
            Fmax = f

    plt.figure()
    plt.plot(fpr, tpr)

    plt.figure()
    plt.plot(recall, precision)

    plt.show()

    auroc = np.zeros((1, thresholds_num - 1)).reshape(-1)
    auprc = np.zeros((1, thresholds_num - 1)).reshape(-1)
    for k in range(thresholds_num - 1):
        auroc[k] = (fpr[k + 1] - fpr[k]) * (tpr[k + 1] + tpr[k]) / 2
        auprc[k] = (recall[k + 1] - recall[k]) * (precision[k + 1] + precision[k]) / 2

    AUROC = np.sum(auroc)
    AUPRC = np.sum(auprc)

    print('AUROC: ', AUROC, '  AUPRC: ', AUPRC, '  Fmax: ', Fmax, '  Smin: ', Smin)

    res_dict = {
        'prediction_score': prediction_score,
        'AUROC': AUROC,
        'AUPRC': AUPRC,
        'Fmax': Fmax,
        'Smin': Smin
    }

    return res_dict
