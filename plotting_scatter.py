import numpy as np
import os
import json
import matplotlib.pyplot as plt

# load the data
arrs = np.load("history3.npy")

print(len(arrs))
print(arrs.shape)

num = 5
ratio = len(arrs)//num if len(arrs) >num else 1

# plt.figure(figsize=(30, 8))
fig, ax = plt.subplots(nrows=1, ncols=num,figsize=(22,6), sharex=True, sharey=True)
fig.tight_layout()
indexes = [0,1,5,15,26]
labels = [0,5,25,75,135]

for i,col in enumerate(ax):
    print(i)
    p = arrs[indexes[i]]
    if i >0:
        p[p==10.0] = 0
    col.plot(list(p), "ro",ms=0.03)
    col.set_xlabel('%dth epochs'%(labels[i]))
    plt.subplots_adjust(left=0.026, bottom=None, right=None, top=None, wspace=0.0, hspace=None)

    if i >0:
        col.yaxis.set_visible(False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.grid(False)
# plt.xlabel("common X")
# plt.ylabel("common Y")
plt.show()