import numpy as np
import platform, os, random
from PIL import Image

def one_hot_encoding(filename):
    label = filename[:filename.index('.')]
    if label == 'cat':
        return 1
    else:
        return 0

def load_data(data_folder, num_training_ratio=0.8, resize=(32, 32)):
    '''
    load all of cats/dogs
    '''
    training_folder = os.path.join(data_folder, "train")
    testing_folder = os.path.join(data_folder, "test")  

    X = []
    Y = []
    for f in os.listdir(training_folder):
        img = Image.open(os.path.join(training_folder, f)).resize(size=resize, resample=Image.ANTIALIAS)
        X.append(np.asarray(img, dtype='float64'))
        Y.append(one_hot_encoding(f))
    X = np.array(X)
    Y = np.array(Y)

    X_te = []
    for f in os.listdir(testing_folder):
        img = Image.open(os.path.join(testing_folder, f)).resize(size=resize, resample=Image.ANTIALIAS)
        X_te.append(np.asarray(img, dtype='float64'))
    X_te = np.array(X_te)

    num_training = int(num_training_ratio * len(X))
    mask = range(num_training)

    X_tr = X[mask]
    Y_tr = Y[mask]

    mask = range(num_training, len(X))
    X_val = X[mask]
    Y_val = Y[mask]

    mean_image = np.mean(X_tr, axis=0)

    X_tr -= mean_image
    X_val -= mean_image

    X_te -= mean_image

    return X_tr, Y_tr, X_val, Y_val, X_te

if __name__ == '__main__':
    load_data('datasets')