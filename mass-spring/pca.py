import pickle
import numpy
from sklearn.decomposition import PCA

import viz

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def main():
    sample_size = 10
    test_size = 100
    data = load_data('mass-spring-bar-configurations_small.pickle')
    test_data = data[:test_size]

    numpy.random.shuffle(data)
    train_data = data[:sample_size]

    pca = PCA(n_components=1)
    pca.fit(train_data)
    
    encoded = pca.transform(test_data)
    decoded = pca.inverse_transform(encoded)

    c = 0
    while True:
        i = c % test_size
        c += 1
        viz.render(decoded[i], [])


if __name__ == '__main__':
    main()
