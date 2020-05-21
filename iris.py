import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle

iris = datasets.load_iris()

s_length = np.double(6.4)

s_width = np.double(3.4)

p_length = np.double(5.1)

p_width = np.double(2.4)

def iris(training_size, test_size, n):
    class_labels = [r'Setosa', r'Versicolor', r'Virginica']
    data, target = datasets.load_iris(return_X_y=True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=1, random_state=42)

    arr1 = [s_length, s_width, p_length, p_width]

    array_try = [arr1]

    array_try = np.array(array_try)

    print(array_try)

    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    array_try = std_scale.transform(array_try)

    print(std_scale)
    print(sample_test)
    print(array_try)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    array_try = pca.transform(array_try)

    print(pca)
    print(sample_test)
    print(array_try)

    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 3)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    array_try = minmax_scale.transform(array_try)

    print(samples)
    print(minmax_scale)
    print(sample_test)
    print(array_try)

    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}

    print(test_input)

    print(f"SEE ARRAY HERE {array_try}")

    return array_try

print(iris(30,10,4))

