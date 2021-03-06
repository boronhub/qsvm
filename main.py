import numpy as np
from qiskit import BasicAer
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.algorithms.many_sample.qsvm import _QSVM_Estimator
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit.aqua import QuantumInstance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def iris(training_size, test_size, n):
    class_labels = [r'Setosa', r'Versicolor', r'Virginica']
    data, target = datasets.load_iris(return_X_y=True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=1, random_state=42)

    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Now reduce number of features to number of qubits
    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 3)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    training_input = {key: (sample_train[label_train == k, :])[:training_size]
                      for k, key in enumerate(class_labels)}
    test_input = {key: (sample_test[label_test == k, :])[:test_size]
                  for k, key in enumerate(class_labels)}
    return sample_train, training_input, test_input, class_labels


def iris_custom(n, s_length, s_width, p_length, p_width ):
    data, target = datasets.load_iris(return_X_y=True)
    sample_train, sample_test, label_train, label_test = \
        train_test_split(data, target, test_size=1, random_state=42)

    arr1 = [s_length, s_width, p_length, p_width]

    array_try = [arr1]

    array_try = np.array(array_try)

    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    array_try = std_scale.transform(array_try)

    pca = PCA(n_components=n).fit(sample_train)
    sample_train = pca.transform(sample_train)
    array_try = pca.transform(array_try)

    samples = np.append(sample_train, array_try, axis=0)
    minmax_scale = MinMaxScaler((-1, 3)).fit(samples)
    array_try = minmax_scale.transform(array_try)

    return array_try


sample_total, training_input, test_input, class_labels = iris(training_size=30,
                                                              test_size=10,
                                                              n=4
                                                              )

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)

seed = 10598

feature_map = SecondOrderExpansion(feature_dimension=4, depth=2, entanglement='linear')
qsvm = QSVM(feature_map, training_input, test_input, multiclass_extension=AllPairs(_QSVM_Estimator, [feature_map]))
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)
result = qsvm.run(quantum_instance)


def run(s_length, s_width, p_length, p_width):
    pred = qsvm.predict(iris_custom(4, s_length, s_width, p_length, p_width), quantum_instance)
    return f"You have found an Iris {class_labels[pred[0]]}!"








