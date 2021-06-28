print(mixture_w.shape)
import numpy as np 
import matplotlib.pyplot as plt
plt.imshow(mixture_w.squeeze().cpu().numpy()[:,200:600])
plt.axis('off')
# plt.set_cmap('hot')
plt.savefig("img.png")
assert 1==2