import numpy as np 
import matplotlib.pyplot as plt 
  
plt.rcParams.update({'font.size': 24})

plt.figure(figsize=(8, 6))

X = ['F1', 'F2', 'F3']
stcs = [0.091, 0.142, 0.174]
s2m2 = [0.084, 0.146, 0.192]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, stcs, 0.4, label = 'STCS')
plt.bar(X_axis + 0.2, s2m2, 0.4, label = 'S2M2')
plt.xticks(X_axis, X)
plt.xlabel("Scenario")
plt.ylabel("Runtime (s)")
plt.ylim([0, 0.3])
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))

X = ['F1', 'F2', 'F3']
stcs = [1.405, 1.447, 1.459]
s2m2 = [1.475, 1.489, 1.521]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, stcs, 0.4, label = 'STCS')
plt.bar(X_axis + 0.2, s2m2, 0.4, label = 'S2M2')
plt.xticks(X_axis, X)
plt.xlabel("Scenario")
plt.ylabel("Ratio")
plt.ylim([0, 2.5])
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))

X = ['S1', 'S2', 'S3']
stcs = [0.145, 0.177, 0.164]
s2m2 = [0.181, 0.153, 0.184]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, stcs, 0.4, label = 'STCS')
plt.bar(X_axis + 0.2, s2m2, 0.4, label = 'S2M2')
plt.xticks(X_axis, X)
plt.xlabel("Scenario")
plt.ylabel("Runtime (s)")
plt.ylim([0, 0.3])
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))

X = ['S1', 'S2', 'S3']
stcs = [1.432, 1.67, 1.586]
s2m2 = [1.545, 1.568, 1.616]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.2, stcs, 0.4, label = 'STCS')
plt.bar(X_axis + 0.2, s2m2, 0.4, label = 'S2M2')
plt.xticks(X_axis, X)
plt.xlabel("Scenario")
plt.ylabel("Ratio")
plt.ylim([0, 2.5])
plt.legend()
plt.show()