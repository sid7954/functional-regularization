import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import sys
import argparse
import os

cuda = torch.device('cuda') 

def save(model, save_dir, save_prefix):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}.pt'.format(save_prefix)
	torch.save(model.state_dict(), save_path)

def dataloader():
	f=open("train.tsv","r")
	g=open("test.tsv","r")
	h=open("unlabelled.tsv","r")
	x_train , y_train , x_test , y_test=[],[],[],[]
	x_unlabelled=[]
	for line in f:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		label=float(parts[1])
		x_train.append(feature)
		y_train.append(label)

	for line in g:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		label=float(parts[1])
		x_test.append(feature)
		y_test.append(label)

	for line in h:
		parts=line.strip("\n").split("\t")
		feature=[float(i) for i in parts[0].split()]
		x_unlabelled.append(feature)

	return x_train , y_train , x_test , y_test, x_unlabelled

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):
        return self.fc(x)

class G_network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(G_network, self).__init__()
        self.input_size = input_size
        self.output_size  = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output = self.fc(x)
        return output

class F_network(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(F_network, self).__init__()
        self.input_size = input_size
        self.output_size  = output_size
        self.fc = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        output=self.fc(x**2)
        return output

def main(args):
	D=args.d
	H=args.r
	train_x , train_y , test_x , test_y, unlabelled_x=dataloader()

	max_num=args.repr_size
	unlabelled_x=unlabelled_x[0:max_num]
	x_unlabelled = torch.FloatTensor(unlabelled_x).cuda()
    
	model = Feedforward(D,H).cuda()
	g_model = G_network(H,D).cuda()
	f_model = F_network(H,1).cuda()
	criterion = torch.nn.MSELoss()
	params = list(model.parameters()) + list(g_model.parameters())
	optimizer = torch.optim.SGD(params, lr=args.repr_lr, momentum=0.9)
    
	model.train()
	g_model.train()
	epoch = args.repr_epochs
	for epoch in range(epoch):
		optimizer.zero_grad()
		y_pred = g_model(model(x_unlabelled))
		regularization_1 = torch.sum(torch.abs( torch.mm(model.fc.weight,model.fc.weight.t()) - torch.eye(H).cuda()))
		regularization_2 = torch.sum(torch.abs( torch.mm(g_model.fc.weight.t(),g_model.fc.weight) - torch.eye(H).cuda()))
		loss = criterion(y_pred.squeeze(), x_unlabelled) + args.reg_w1*regularization_1 + args.reg_w2*regularization_2
		if (epoch % 1000) == 0:
			print('Epoch {}: auto-encoder train loss: {}'.format(epoch, loss.item()))
		loss.backward()
		optimizer.step()
	save(model, "saved", 'model_r=30')
	# model.load_state_dict(torch.load("saved/model_r=30.pt", map_location=torch.device('cpu')))
	if args.repr_static:
		params2 = list(f_model.parameters())
		for param in model.parameters():
			param.requires_grad = False
	else:
		params2 = list(f_model.parameters()) + list(model.parameters())
	optimizer2 = torch.optim.SGD(params2, lr=args.pred_lr, momentum=0.9)

	max_num=args.pred_size
	train_x=train_x[0:max_num]
	train_y=train_y[0:max_num]
	x_train = torch.FloatTensor(train_x).cuda()
	y_train = torch.FloatTensor(train_y).cuda()
	x_test = torch.FloatTensor(test_x).cuda()
	y_test = torch.FloatTensor(test_y).cuda()

	model.train()
	f_model.train()
	epoch = args.pred_epochs
	best_loss=(10000*torch.ones(1)).cuda()
	best_predictions=(1000*torch.ones(1)).cuda()
	for epoch in range(epoch):
		optimizer2.zero_grad()
		y_pred = f_model(model(x_train))
		loss = criterion(y_pred.squeeze(), y_train)
		if (epoch % 500) == 0:
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
	argparser.add_argument("--repr_epochs", type=int, default=20000)
	argparser.add_argument("--pred_epochs", type=int, default=20000)
	argparser.add_argument("--repr_size", type=int, default=10000)
	argparser.add_argument("--pred_size", type=int, default=5000)
	argparser.add_argument("--num_trials", type=int, default=1)
	argparser.add_argument("--reg_w1", type=float, default=0.1)
	argparser.add_argument("--reg_w2", type=float, default=0.01)
	argparser.add_argument("--repr_lr", type=float, default=0.001)
	argparser.add_argument("--pred_lr", type=float, default=0.000001)
	argparser.add_argument("--repr_static", action='store_true', default=False)
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