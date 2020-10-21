import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import sys
import argparse

cuda = torch.device('cuda') 

def dataloader():
	f=open("train.tsv","r")
	g=open("test.tsv","r")
	x_train , x_train_mask, y_train , x_test, x_test_mask , y_test=[],[],[],[],[],[]
	for line in f:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		masked=float(parts[1])
		label=float(parts[2])
		x_train.append(feature)
		x_train_mask.append(masked)
		y_train.append(label)

	for line in g:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		masked=float(parts[1])
		label=float(parts[2])
		x_test.append(feature)
		x_test_mask.append(masked)
		y_test.append(label)

	return x_train , x_train_mask, y_train , x_test, x_test_mask , y_test

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):
        return self.fc(x)**2

class F_network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(F_network, self).__init__()
        self.input_size = input_size
        self.output_size  = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output=self.fc(x)
        return output

def main(args):
	D=args.d
	H=args.r
	x_train , x_train_mask, y_train , x_test, x_test_mask , y_test=dataloader()

	max_num=args.pred_size
	x_train=x_train[0:max_num]
	x_train_mask=x_train_mask[0:max_num]
	y_train=y_train[0:max_num]

	x_train = torch.FloatTensor(x_train).cuda()
	x_train_mask = torch.FloatTensor(x_train_mask).cuda()
	y_train = torch.FloatTensor(y_train).cuda()
	x_test = torch.FloatTensor(x_test).cuda()
	x_test_mask = torch.FloatTensor(x_test_mask).cuda()
	y_test = torch.FloatTensor(y_test).cuda()

	model = Feedforward(D,H).cuda()
	f_model = F_network(H,1).cuda()
	criterion = torch.nn.MSELoss()
	params = list(model.parameters()) + list(f_model.parameters())
	optimizer2 = torch.optim.SGD(params, lr=args.pred_lr, momentum=0.9)

	model.train()
	f_model.train()
	epoch = args.pred_epochs
	best_loss=(10000*torch.ones(1)).cuda()
	best_predictions=(1000*torch.ones(1)).cuda()
	for epoch in range(epoch):
		optimizer2.zero_grad()
		y_pred = f_model(model(x_train))
		loss = criterion(y_pred.squeeze(), y_train)           
		if (epoch % 100) == 0:
			print('Epoch {}: regression train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer2.step()
		
		if (epoch % 500) == 0:
			model.eval()
			f_model.eval()
			y_pred = f_model(model(x_test))
			loss = criterion(y_pred.squeeze(), y_test) 
			print('TEST Regression loss: {}'.format(loss.item()))
			model.train()
			f_model.train()
			if loss < best_loss:
				best_loss=loss
				best_predictions=y_pred.squeeze()
	return best_loss.item(), best_predictions.cpu().data.numpy()
if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
	argparser.add_argument("--pred_epochs", type=int, default=10000)
	argparser.add_argument("--pred_size", type=int, default=5000)
	argparser.add_argument("--pred_lr", type=float, default=0.00001)
	argparser.add_argument("--num_trials", type=int, default=1000)
	argparser.add_argument("--r", type=int, default=30)
	argparser.add_argument("--d", type=int, default=100)
	args = argparser.parse_args()
	total_loss=0
	all_predictions=np.zeros([args.num_trials,1000])
	for i in range(args.num_trials):
		print(i)
		sys.stdout.flush()
		epoch_loss, epoch_predictions=main(args)
		total_loss+=epoch_loss
		all_predictions[i,:]=epoch_predictions
	np.save('d_100_r_30.npy',all_predictions) 
	print(total_loss/args.num_trials)