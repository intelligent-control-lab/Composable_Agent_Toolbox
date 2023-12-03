import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['F', 'S1', 'S2']
stsc = [0.174, 0.145, 0.177]
s2m2 = [0.192, 0.181, 0.153]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, stsc, 0.4, label = 'STCS')
plt.bar(X_axis + 0.2, s2m2, 0.4, label = 'S2M2')

plt.xticks(X_axis, X)
plt.xlabel("Scenario")
plt.ylabel("Runtime (s)")
plt.ylim([0, 0.25])
plt.legend()
plt.show()