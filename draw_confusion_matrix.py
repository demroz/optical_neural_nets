import numpy as np
import matplotlib.pyplot as plt

mat = np.loadtxt('confusion_matrix.csv')

for i in range(0,10):
    norm_fact = 0
    for j in range(0,10):
        norm_fact += mat[i,j]

    for j in range(0,10):
        mat[i,j] /= norm_fact

mat = mat*100
fig, ax = plt.subplots()
im = ax.imshow(mat, interpolation = 'nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

classes = ['0','1','2','3','4','5','6','7','8','9']
title = 'mnist'
ax.set(xticks=np.arange(mat.shape[1]),
           yticks=np.arange(mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
fmt = '.2f'
thresh = mat.max() / 2.
for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black")

plt.show()
