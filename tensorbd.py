import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

dt_string = datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")

writer = SummaryWriter('logs/'+ dt_string)

x = 3

for step in range(1, 1001):
	writer.add_scalar('x', x, step)
	writer.add_scalar('y', x*2, step)
	x +=1
	writer.add_scalars('sin and cos', {'sin': x, 'cos': x*2}, step)
writer.close()