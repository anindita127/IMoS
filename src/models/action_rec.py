import torch
import numpy as np
import torch.nn as nn
import os
import sys
sys.path.append('.')
sys.path.append('..')
import itertools
import smplx
import src.tools.utils as data_utils
import torch.nn.functional as F
import torch.autograd as autograd

class LSTM_Action_Classifier(nn.Module):
	def __init__(self, joints_dim=44, hidden_dim=128, label_size=29, batch_size=1, num_layers=2, kernel_size=3):   #LSTMClassifier(48, 128, 8, 1, 2, 3)
		super(LSTM_Action_Classifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		joints_dim2d = joints_dim 
		
		self.lstm2_2 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv1_2_2 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		self.lstm2_3 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv1_2_3 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		self.sig = nn.Sigmoid()
		self.hidden2_2 = self.init_hidden2_2()
		self.hidden2_3 = self.init_hidden2_3()
		
		self.hidden2label = nn.Linear(hidden_dim, label_size)
	
	def init_hidden2_1(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	def init_hidden2_2(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	def init_hidden2_3(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	
	def predict(self, joints3d_vec):
		x3 = joints3d_vec
		x2 = x3.view(-1, 22, 3)
		x2_2 = x2[:,:,0:2].contiguous().view(-1, 1, 44)
		x2_3 = x2[:,:,[0,2]].contiguous().view(-1, 1, 44)
		lstm_out2_2, self.hidden2_2_ = self.lstm2_2(x2_2, self.hidden2_2)
		lstm_out2_3, self.hidden2_3_ = self.lstm2_3(x2_3, self.hidden2_3)
		t2_2 = lstm_out2_2[-1].view(self.batch_size,1,-1)
		t2_3 = lstm_out2_3[-1].view(self.batch_size,1,-1)
		y2_2 = self.conv1_2_2(t2_2)
		y2_3 = self.conv1_2_3(t2_3)
		y3 = y2_2+y2_3
		y3 = y3.contiguous().view(-1, self.hidden_dim)        
		y4  = self.hidden2label(y3)
		y_pred = self.sig(y4)
		return  y_pred, torch.tanh(y3)*0.1


	def forward(self, joints3d_vec, y):
		x3 = joints3d_vec
		x2 = x3.view(-1, 22, 3)
		x2_2 = x2[:,:,0:2].contiguous().view(-1, 1, 44)
		x2_3 = x2[:,:,[0,2]].contiguous().view(-1, 1, 44)
		lstm_out2_2, self.hidden2_2_ = self.lstm2_2(x2_2, self.hidden2_2)
		lstm_out2_3, self.hidden2_3_ = self.lstm2_3(x2_3, self.hidden2_3)
		t2_2 = lstm_out2_2[-1].view(self.batch_size,1,-1)
		t2_3 = lstm_out2_3[-1].view(self.batch_size,1,-1)
		y2_2 = self.conv1_2_2(t2_2)
		y2_3 = self.conv1_2_3(t2_3)
		y3 = y2_2+y2_3
		y3 = y3.contiguous().view(-1, self.hidden_dim)        
		y4  = self.hidden2label(y3)
		y_pred = self.sig(y4)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred