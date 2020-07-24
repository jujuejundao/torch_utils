import os
from datetime import datetime
import argparse
import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(args):
	torch.manual_seed(0)

	args.world_size = 1
	if args.distributed:
	    torch.cuda.set_device(args.local_rank)
	    torch.distributed.init_process_group(backend='nccl',
	                                         init_method='env://')
	    args.world_size = torch.distributed.get_world_size()

	assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

	torch.backends.cudnn.benchmark = True

	model = ConvNet().cuda()
	optimizer = torch.optim.SGD(model.parameters(), 1e-4)

	model, optimizer = amp.initialize(model, optimizer,
										opt_level=args.opt_level,
										keep_batchnorm_fp32=args.keep_batchnorm_fp32
										)



	if args.distributed:
		model = DDP(model, delay_allreduce=True)

	loss_fn = nn.CrossEntropyLoss().cuda()

	# Data loading code
	train_dataset = torchvision.datasets.MNIST(root='./data',
	                                           train=True,
	                                           transform=transforms.ToTensor(),
	                                           download=True)

	train_sampler = torch.utils.data.distributed.DistributedSampler(
		train_dataset,
		num_replicas=int(os.environ['WORLD_SIZE']),
		rank=args.local_rank
	)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=args.batch_size,
	                                           shuffle=False,
	                                           num_workers=0,
	                                           pin_memory=True,
	                                           sampler=train_sampler)

	start = datetime.now()
	total_step = len(train_loader)
	for epoch in tqdm.trange(args.epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = images.cuda(non_blocking=True)
			labels = labels.cuda(non_blocking=True)
			outputs = model(images)
			optimizer.zero_grad()
			loss = loss_fn(outputs, labels)
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()
			optimizer.step()
			if (i + 1) % 100 == 0 and args.local_rank == 0:
				print('Rank {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
					args.local_rank,
					epoch + 1, 
					args.epochs, 
					i + 1, 
					total_step,
					loss.item())
					)
		print(loss.data)

		# Collect loss from all processes
		reduced_loss = reduce_tensor(loss.data)

		if args.local_rank == 0:
		    print("Training complete in: " + str(datetime.now() - start))
		    print(reduced_loss)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= 2
    return rt

def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('-n', '--nodes', default=2, type=int, metavar='N')
	# parser.add_argument('-g', '--gpus', default=1, type=int,
	#                     help='number of gpus per node')
	parser.add_argument('-bz', '--batch_size', default=64, type=int,
	                    help='batch size')
	parser.add_argument('--epochs', default=2, type=int, metavar='N',
	                    help='number of total epochs to run')
	parser.add_argument("--local_rank", default=0, type=int)
	parser.add_argument('--opt-level', default='O1', type=str)
	parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
	args = parser.parse_args()

	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1

	train(args)

if __name__ == '__main__':
    main()