import numpy as np
from scipy import special
import json
import os
#t = np.loadtxt("result2.txt",delimiter="[")
#print(t)
def your_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
np.set_printoptions(threshold=np.inf)
before = "output/before/pattern_losses.txt"
after = "result.txt"

f = open("avgloss_result1.txt", "r")
txt = f.read()
txt = txt[1:-1]
txt = np.array([float(x) for x in txt.split(", ")])

f1 = open("avgloss_result3.txt", "r")
txt1 = f1.read()
txt1 = txt1[1:-1]
txt1 = np.array([float(x) for x in txt1.split(", ")])
#txt1 = txt1[txt1!=10.0]

f2 = open("output/after/pattern_losses.txt", "r")
txt2 = f2.read()
txt2 = txt2[1:-1]
txt2 = np.array([float(x) for x in txt2.split(", ")])


'''
txt_sm = special.softmax(txt)
print(txt_sm)
with open("output/before/pattern_dist.txt","w") as f:
    f.write(json.dumps(list(txt_sm)))
print(txt_sm.sum())
'''
#print(txt[0:1000])


#print(np.where(txt==float(10.0)))
print("avgloss_result1")
print(np.histogram(txt,bins=100))
#print(np.nonzero(np.histogram(txt,bins=100)[0])[0])
hist,bins = np.histogram(txt,bins=100)
#vals = np.digitize(txt,bins)
#print(txt[vals==35])
print("avgloss_result3")
print(np.histogram(txt1,bins=100))
hist1,bins1 = np.histogram(txt1,bins=100)
counts = np.load("counts3.npy",allow_pickle=True)
vals1 = np.digitize(txt1,bins1)
bad_patterns_ind = np.where((vals1>=2)&(vals1<99))
bad_patterns = txt1[bad_patterns_ind]
print(bad_patterns)
#print(bad_patterns_ind)
counts = counts.item()
print(type(counts))
inds = np.where((vals1>=0)&(vals1<2))
#print(inds)
#for i in inds[0]:
#    print("pattern id = ",i)
#    print(counts[i])
total = 0
for each in counts:
    total += counts[each]
print("avg:",total/len(counts))
for i in bad_patterns_ind[0]:
    print("pattern id = ",i)
    print(counts[i])


'''
mask_idxs_exclude = []
mask_idxs_include = []
if os.path.isfile("output/before/pattern_losses.txt"):
    f = open("output/before/pattern_losses.txt", "r")
    old = f.read()
    old = old[1:-1]
    old = np.array([float(x) for x in old.split(", ")])
    mask_idxs_exclude = np.where(old==10.0)[0]
    mask_idxs_include = np.where(old!=10.0)[0]
#print(np.histogram(txt[mask_idxs_include],bins=100))
txt = txt[:30395]
print(sum(np.bitwise_and((txt==10.0),(txt2==10.0))))
'''
