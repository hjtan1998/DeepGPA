#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Haojiang_Tan
# Datetime： 2021/8/16 20:44
# IDE： PyCharm


import argparse

def read_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train_iter_n', type=int, default=150, help = 'max number of training iteration')

	parser.add_argument('--batch_size', type=int, default=512, help = 'batch_size')

	parser.add_argument('--ini_d', type=int, default=128, help = 'initialized dimension')

	parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')

	parser.add_argument('-Layers', action='store', dest='Layers', default={
		'Layer_1': [64, 32],
		'Layer_2': [64, 32],
		'Layer_3': [64, 32],
		'Layer_4': [64, 32]
	})

	args = parser.parse_args()

	return args