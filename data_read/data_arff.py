import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


def transform_arff_data(data):
    X = []
    y = []
    for sample in data:
        x = []
        for id, value in enumerate(sample):
            if id == len(sample) - 1:
                y.append(value)
            else:
                x.append(value)
        X.append(x)


    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)


def create_2d4c():
    data, meta = arff.loadarff('./data/2d-4c-no4.arff')
    return transform_arff_data(data)

def create_2d10c():
    data, meta = arff.loadarff('./data/2d-10c.arff')
    return transform_arff_data(data)

def create_2d20c():
    data, meta = arff.loadarff('./data/2d-20c-no0.arff')
    return transform_arff_data(data)

def create_3spiral():
    data, meta = arff.loadarff('./data/3-spiral.arff')
    return transform_arff_data(data)

def create_aggregation():
    data, meta = arff.loadarff('./data/aggregation.arff')
    return transform_arff_data(data)

def create_compound():
    data, meta = arff.loadarff('./data/compound.arff')
    return transform_arff_data(data)

def create_elly_2d10c13s():
    data, meta = arff.loadarff('./data/elly-2d10c13s.arff')
    return transform_arff_data(data)

def create_s1():
    data, meta = arff.loadarff('./data/s-set1.arff')
    return transform_arff_data(data)

def create_s2():
    data, meta = arff.loadarff('./data/s-set2.arff')
    return transform_arff_data(data)