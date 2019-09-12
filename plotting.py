import numpy as np
import os
import json
import matplotlib.pyplot as plt

# load the data
f = open("data/test_mb/avgloss_result3.txt")
txt = f.read()
txt = txt[1:-1]
txt = np.array([float(x) for x in txt.split(", ")])
txt = txt[txt!=10.0]
# print(np.histogram(txt,bins=100))

f2 = open("data/test_mb/pattern_losses.txt")
txt1 = f2.read()
txt1 = txt1[1:-1]
txt1 = np.array([float(x) for x in txt1.split(", ")])
txt1 = txt1[txt1!=10.0]

mask_idxs_exclude = []
mask_idxs_include = []
# mask_idxs_include = np.where(txt!=0.0)
# if os.path.isfile("data/before/pattern_losses.txt"):
#     f = open("data/before/pattern_losses.txt", "r")
#     old = f.read()
#     old = old[1:-1]
#     old = np.array([float(x) for x in old.split(", ")])
#     mask_idxs_exclude = np.where(old==10.0)[0]
#     mask_idxs_include = np.where(old!=10.0)[0]

print(np.mean(txt))
print(np.histogram(txt,bins=100)[0])
plt.figure(figsize=(18, 10))
plt.subplot(211)

plt.hist(np.log(txt),bins=100)
plt.xlabel('Log Loss')
plt.ylabel('Quantity')

# plt.xlabel('Patterns')
# plt.title("Mini-Batch")
# plt.plot(list(txt1), "ro",ms=0.1)
# plt.ylabel('Loss')
# axes = plt.gca()
# axes.set_ylim([0,10])



plt.subplot(212)
# plt.title("Normal")
plt.plot(list(txt), "ro",ms=0.03)
print(np.where(txt>2.0))
plt.xlabel('Patterns')
plt.ylabel('Loss')
axes = plt.gca()
axes.set_ylim([-0.5,10.5])
plt.show()