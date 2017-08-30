import numpy as np
import platform, os, random,bcolz
from PIL import Image

def one_hot_encoding(filename):
    label = filename[:filename.index('.')]
    if label == 'cat':
        return 1
    else:
        return 0

def save_array(data_folder, fname, arr):
    fname = os.path.join(data_folder, fname)
    print("Saving to {0} ...".format(fname))
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(data_folder, fname):
    fname = os.path.join(data_folder, fname)
    print("Loading from {0} ...".format(fname))
    return bcolz.open(fname)[:]

def load_from_cache(data_folder, size):
    X_tr = load_array(data_folder, 'train{0}_data.npz'.format(size))
    Y_tr = load_array(data_folder, 'train{0}_label.npz'.format(size))
    X_val = load_array(data_folder, 'val{0}_data.npz'.format(size))
    Y_val = load_array(data_folder, 'val{0}_label.npz'.format(size))

    X_te = load_array(data_folder, 'test{0}.npz'.format(size))

    return X_tr, Y_tr, X_val, Y_val, X_te

def save_to_cache(data_folder, size, X_tr, Y_tr, X_val, Y_val, X_te):
    save_array(data_folder, 'train{0}_data.npz'.format(size), X_tr)
    save_array(data_folder, 'train{0}_label.npz'.format(size), Y_tr)
    save_array(data_folder, 'val{0}_data.npz'.format(size), X_val)
    save_array(data_folder, 'val{0}_label.npz'.format(size), Y_val)
    save_array(data_folder, 'test{0}.npz'.format(size), X_te)

def load_data(data_folder, num_training_ratio=0.8, resize=(32, 32)):
    '''
    load all of cats/dogs
    '''
    try:
        return load_from_cache(data_folder, size=resize[0])
    except FileNotFoundError:
        print("Cache not found, loading from raw images...")
        pass

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

    save_to_cache(data_folder, resize[0],  X_tr, Y_tr, X_val, Y_val, X_te)

    return X_tr, Y_tr, X_val, Y_val, X_te

if __name__ == '__main__':
    load_data('datasets', resize=(64,64))