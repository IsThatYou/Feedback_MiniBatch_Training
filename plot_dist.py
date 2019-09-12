import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

number_of_patterns = 27066
pattern_dist = np.zeros(number_of_patterns)
# norm_dist = scipy.stats.norm(number_of_patterns//2,5000)
norm_dist = scipy.stats.norm(0,5000)

for i in range(number_of_patterns):
    pattern_dist[i] = norm_dist.pdf(i)
# pattern_dist = softmax(np.zeros(number_of_patterns)).tolist()

plt.figure(figsize=(18, 10))
plt.plot(pattern_dist)
plt.show()