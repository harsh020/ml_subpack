from dimensionality_reduction import PCA

from scipy import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image

# mat = io.loadmat('bird_small.mat')
# X = mat['A']

img = Image.open('bird_small.png')

X = np.asarray(img)

m = X.shape[0]
n = X.shape[1]
X = X.reshape(m*n, 3)

image = Image.fromarray(X)
pca = PCA(k_dim=2)
pca.fit(X[:100, :])
Xt = pca.transform(X[:100, :])
print(Xt.shape)
plt.scatter(Xt[:, 0], Xt[:, 1])

plt.show()
