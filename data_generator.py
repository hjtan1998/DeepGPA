#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Haojiang_Tan
# Datetime： 2021/8/7 19:21 
# IDE： PyCharm

import numpy as np

class input_data(object):

	def __init__(self):

		# R_2_2 = np.load('./data/miRNA447_miRNA447.npy')
		# R_4_4 = np.load('./data/TO159_TO159.npy')
		# R_1_2 = np.load('./data/gene39296_miRNA447.npy')
		# R_1_3 = np.load('./data/gene39296_pathway465.npy')
		# R_1_4 = np.load('./data/gene39296_TO159.npy')
		# R_2_4 = np.load('./data/miRNA447_TO159.npy')
		# X_1_1 = np.load('./data/gene39296_sep_per.npy')
		# X_1_2 = np.load('./data/gene39296_go_pca_512.npy')
		# X_4_1 = np.load('./data/TO159_emb_feature.npy')

		R_2_2 = np.load('./data/miRNA447_miRNA447.npy')
		R_4_4 = np.load('./data/TO159_TO159.npy')
		R_1_2 = np.load('./data/gene2000_miRNA447.npy')
		R_1_3 = np.load('./data/gene2000_pathway465.npy')
		R_1_4 = np.load('./data/gene2000_TO159.npy')
		R_2_4 = np.load('./data/miRNA447_TO159.npy')
		X_1_1 = np.load('./data/gene2000_sep_per.npy')
		X_1_2 = np.load('./data/gene2000_go_pca_512.npy')
		X_4_1 = np.load('./data/TO159_emb_feature.npy')

		self.R_2_2 = R_2_2
		self.R_4_4 = R_4_4
		self.R_1_2 = R_1_2
		self.R_1_3 = R_1_3
		self.R_1_4 = R_1_4
		self.R_2_4 = R_2_4

		X_1_1 = (X_1_1 - np.min(X_1_1))/(np.max(X_1_1) - np.min(X_1_1))
		X_1_2 = (X_1_2 - np.min(X_1_2))/(np.max(X_1_2) - np.min(X_1_2))
		X_4_1 = (X_4_1 - np.min(X_4_1))/(np.max(X_4_1) - np.min(X_4_1))

		self.X_1_1 = X_1_1
		self.X_1_2 = X_1_2
		self.X_4_1 = X_4_1

















