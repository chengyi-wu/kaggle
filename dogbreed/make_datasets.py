import csv
import os
import random

def get_classes(filename):
    ht = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            if i == 0:
                i += 1
                continue
            id, breed = line[0], line[1]
            if breed not in ht:
                ht[breed] = []
            ht[breed].append(id)
    classes = list(ht.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx, ht

def make_datasets():
    filename = os.path.join('datasets', 'labels.csv')
    classes, class_to_idx, ht = get_classes(filename)
    raw_dir = os.path.join(os.getcwd(), 'datasets', 'raw')
    files = {f[:f.index('.')]:os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir))}
    train_dir = os.path.join(os.getcwd(), 'datasets', 'train')
    val_dir = os.path.join(os.getcwd(), 'datasets', 'val')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    for class_name in classes:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(train_class_dir):
            os.mkdir(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.mkdir(val_class_dir)
        random.shuffle(ht[class_name])
        for i, filename in enumerate(ht[class_name]):
            f = files[filename]
            ext = f[f.index('.'):]
            if i < len(ht[class_name]) * .8:
                target = os.path.join(train_class_dir, str(i) + ext)
            else:
                target = os.path.join(val_class_dir, str(i) + ext)
            if not os.path.exists(target):
                os.symlink(f, target)
            print("%s => %s" % (f, target))
                
if __name__ == '__main__':
    make_datasets()
