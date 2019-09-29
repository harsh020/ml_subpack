import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def disp_imgs(X, no, l_size):
    idx = np.random.randint(0, X.shape[0], no)
    rows = int(np.sqrt(no))
    fig, sub = plt.subplots(rows, rows, sharex=True, sharey=True)

    X_ = []
    for img in X[idx, :]:
        X_.append(np.array(img)*225)
    X_ = np.array(X_)

    for i, ax in zip(range(len(X_)), sub.ravel()):
        size = int(np.sqrt(l_size))
        img = Image.fromarray(np.transpose(X_[i, :].reshape((size, size))))
        ax.imshow(img)

    plt.show()
