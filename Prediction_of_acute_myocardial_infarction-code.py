from cmath import isnan
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA

OUTPUT_FOLDER = "xxx"
DATA_PATH = "xxx.xlsx"
BATCH_SIZE = 16
EPOCHS = 100
EARLY_STOP = True
SAVE_MODEL = True
HISTORY_FOLDER = os.path.join(OUTPUT_FOLDER,"history")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "models")
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
METRICS = [keras.metrics.BinaryAccuracy(name='accuracy')]
PERCENT = 0.2

def plot_history_2(history, save_path=None, show=True):
    plt.figure(dpi=800)

    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(1, 2, n+1)
        plt.plot(history.epoch,
                 history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

        plt.legend()
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close("all")

def make_model(layers):
    """
    To built a fully connected neural network  according to the number of nodes in each layer
    Args:
        layers (list): [n0, n1, ...], Number of nodes in the hidden layer

    Returns:
        model: keras.Sequential()
    """
    model = keras.Sequential()
    for layer in layers:
        model.add(keras.layers.Dense(layer, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

class HNN_layer(keras.layers.Layer):
    def __init__(self, feature_list=[10, 13, 13, 7, 3, 6, 3, 13, 5, 4, 6, 4, 5, 4, 4, 5, 4, 3, 3, 7, 3, 3], units=1, activation="relu", **kwargs):
        self._feature_list = feature_list
        self.units = units
        self.activation = keras.layers.Activation(activation)
        self.final_activation = keras.layers.Activation("sigmoid")
        super(HNN_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        """The parameters needed to build the model"""
        self.kernel_list = []
        self.bias_list = []
        for i, feature_num in enumerate(self._feature_list):
            kernel = self.add_weight(name="kernel"+str(i),
                                          shape=(feature_num, self.units),
                                          initializer="uniform",
                                          trainable=True)
            self.kernel_list.append(kernel)

            bias = self.add_weight(name="bias"+str(i),
                                        shape=(self.units,),
                                        initializer="zeros",
                                        trainable=True)
            self.bias_list.append(bias)
        self.final_kernel = self.add_weight(name="final_kernel",
                                            shape=(
                                                len(self.kernel_list), self.units),
                                            initializer="uniform", trainable=True)
        self.final_bias = self.add_weight(name="final_bias",
                                          shape=(self.units,),
                                          initializer="zeros", trainable=True)
        super(HNN_layer, self).build(input_shape)

    def call(self, x):
        """Forward calculation"""
        nodes = []
        start_i = 0
        for i, feature_num in enumerate(self._feature_list):
            kernel, bias = self.kernel_list[i], self.bias_list[i]
            node = self.activation(
                x[:, start_i:start_i+feature_num]@kernel+bias)
            nodes.append(node)
        nodes_tensor = tf.concat(nodes, 1)
        return self.final_activation(nodes_tensor@self.final_kernel+self.final_bias)

    def get_config(self):
        config = super().get_config().copy()
        return config


class Classifier:
    def __init__(self, data_path) -> None:
        """Data initialization, data reading, data set partitioning and standardization"""
        self._data_path = data_path
        self._data = None
        self._feature_names = None
        self.METHOD_FNN = 0
        self.METHOD_PCA = 1
        self.METHOD_HNN = 2
        self.get_data(data_path)

    def standardaize(self, data) -> np.ndarray:
        """data standardization

        Args:
            data (np.ndarray): 

        Returns:
            np.ndarray: 
        """
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def _prepare_data(self, data) -> tuple:
        """Data set partitioning, data standardization, printing data set information

        Args:
            data (ndarray): 

        Returns:
            tuple: train_featrues, train_labels, val_features, val_labels, test_features, test_labels
        """
        train_df, test_df = train_test_split(data, test_size=PERCENT)
        train_df, val_df = train_test_split(train_df, test_size=PERCENT)

        # Form np arrays of labels and features.
        train_labels = train_df[:, 0]
        val_labels = val_df[:, 0]
        test_labels = test_df[:, 0]

        train_features = train_df[:, 1:]
        val_features = val_df[:, 1:]
        test_features = test_df[:, 1:]

        scaler = StandardScaler()

        train_features = scaler.fit_transform(train_features)

        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        print('Training labels shape:', train_labels.shape)
        print('Validation labels shape:', val_labels.shape)
        print('Test labels shape:', test_labels.shape)

        print('Training features shape:', train_features.shape)
        print('Validation features shape:', val_features.shape)
        print('Test features shape:', test_features.shape)

        print("Training 1 in all:", np.sum(
            train_labels)/train_labels.shape[0])
        print("Validation 1 in all:", np.sum(
            val_labels)/val_labels.shape[0])
        print("Test 1 in all:", np.sum(
            test_labels)/test_labels.shape[0])

        return train_features, train_labels, val_features, val_labels, test_features, test_labels

    def get_data_path(self) -> str:
        return self._data_path

    def get_feature_names(self) -> dict:
        return self._feature_names

    def get_data(self, data_path=None) -> np.ndarray:
        """data reading

        Args:
            data_path (str): excel-path
        return:
            ndarray: 
        """
        def is_feature(s):
            if s[0:3] == "Myo" or s[0:3] == "Con":
                return False
            for i in range(3):
                if not s[i].isalnum() and s[i] != ' ' and s[i] != '-':
                    return False
            return True

        def is_small_feature(s):
            for i in range(3):
                if not s[i].isupper() and not s[i].isdigit():
                    return False
            return True

        
        if self._data is None:
            data_df = pd.read_excel(data_path)
            data = []
            self._feature_names = {}
            feature_group = None
            
            for row_df in data_df.values:
                row = []
                for cell in row_df:
                    # data
                    if isinstance(cell, float) and not isnan(cell):
                        row.append(cell)
                    # feature name
                    elif isinstance(cell, str) and is_feature(cell):
                        if is_small_feature(cell):
                            self._feature_names[feature_group].append(cell)
                        else:
                            feature_group = cell
                            self._feature_names[feature_group] = []
                if len(row) != 0:
                    data.append(row)
            data = np.array(data)
            data = np.insert(data, 0, np.zeros((1, data.shape[1])), axis=0)
            data[0, :49] = 1
            self._data = data.T

        return self._data

    def run(self, method, history_name=None, model_name=None) -> list:
        if method == self.METHOD_FNN:
            if history_name is None:
                history_name = "fnn.png"
            return self.run_fnn(self._data, history_name, model_name)
        elif method == self.METHOD_PCA:
            if history_name is None:
                history_name = "pca+fnn.png"
            return self.run_pca(history_name, model_name)
        elif method == self.METHOD_HNN:
            if history_name is None:
                history_name = "hnn.png"
            return self.run_hnn(history_name, model_name)
        else:
            raise("wrong method!")

    def run_fnn(self, datas, history_name, model_name=None) -> list:
        """Fully connected neural networks

        Args:
            datas (ndarray): 
            history_name (str): history.png

        Returns:
            list: [loss, accuracy]
        """
        
        train_features, train_labels, val_features, val_labels, test_features, test_labels = self._prepare_data(
            datas)
        # To build the network
        model = make_model([])
        # Optimizer and loss function
        model.compile(keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=METRICS)
        # EARLY_STOP
        if EARLY_STOP:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=10,
                mode='auto',
                restore_best_weights=True)
            callbacks = [early_stopping]
        else:
            callbacks = None
        # training
        history = model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=(val_features, val_labels),)
        if model_name is not None:
            model.save(os.path.join(MODEL_FOLDER, model_name))
        
        predictions = model.predict(
            test_features, batch_size=BATCH_SIZE)
        # save history
        history_df = pd.DataFrame(
            history.history, index=history.epoch)
        history_df.to_excel(os.path.join(
            HISTORY_FOLDER, model_name+".xlsx"))
        
        plot_history_2(history, os.path.join(
            IMAGE_FOLDER, history_name), False)
        # evaluate
        results = model.evaluate(test_features, test_labels,
                                 batch_size=BATCH_SIZE, verbose=0)
        return results

    def pca_transform(self, raw_datas, reference_datas=None, first_col=False) -> np.ndarray:
        """Dimensionality reduction in PCA

        Args:
            raw_data (ndarray): 
            reference_data (ndarray): 
            first_col (bool): 

        Returns:
            np.ndarray: 
        """
        if reference_datas is None:
            reference_datas = self._data
        # Data Standardization
        reference_datas = reference_datas.copy()
        reference_datas[:, 1:] = self.standardaize(reference_datas[:, 1:])
        
        start_col_refe = 1  
        start_col_data = int(first_col) 
        variace_datios = []
        singular_values = []
        new_datas = np.reshape(raw_datas[:, 0], (raw_datas.shape[0], 1))
        for feature_group in self._feature_names.keys():
            
            feature_num = len(self._feature_names[feature_group])
            # pca
            pca = PCA(n_components=1)
            pca.fit(
                reference_datas[:, start_col_refe:start_col_refe+feature_num])
            new_data = pca.transform(
                raw_datas[:, start_col_data:start_col_data+feature_num])
            variace_datios.append(pca.explained_variance_ratio_[0])
            singular_values.append(pca.singular_values_[0])
            new_datas = np.hstack((new_datas, new_data))
            
            start_col_refe = start_col_refe+feature_num
            start_col_data = start_col_data+feature_num
        if not first_col:
            new_datas=np.delete(new_datas, 0, axis=1)
        print(f"PCA:{raw_datas.shape} -> {new_datas.shape}")
        return new_datas

    def run_pca(self, history_name, model_name) -> list:
        """PCA + FNN
        PCA：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        Returns:
            list: [loss, accuracy]
        """
        # Data dimension reduction
        new_datas = self.pca_transform(self._data, self._data, True)
        
        return self.run_fnn(new_datas, history_name, model_name)

    def run_hnn(self, history_name, model_name=None) -> list:
        """HNN

        Returns:
            list: [loss, accuracy]
        """
        feature_list = []
        for feature_group in self._feature_names:
            feature_list.append(len(self._feature_names[feature_group]))
        print(f"feature_list:{feature_list}")
        # DATA
        train_features, train_labels, val_features, val_labels, test_features, test_labels = self._prepare_data(
            self._data)
        # To build the network
        model = keras.Sequential(HNN_layer(feature_list))
        # Optimizer and loss function
        model.compile(keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=METRICS)
        # EARLY_STOP
        if EARLY_STOP:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=10,
                mode='auto',
                restore_best_weights=True)
            callbacks = [early_stopping]
        else:
            callbacks = None
        # training
        history = model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=(val_features, val_labels),)
        if model_name is not None:
            model.save(os.path.join(MODEL_FOLDER, model_name))
        # prediction
        predictions = model.predict(
            test_features, batch_size=BATCH_SIZE)
        # save history
        history_df = pd.DataFrame(
            history.history, index=history.epoch)
        history_df.to_excel(os.path.join(
            HISTORY_FOLDER, model_name+".xlsx"))
        """
        plot_history_2(history, os.path.join(
            IMAGE_FOLDER, history_name), False)
        """
        # evaluate
        results = model.evaluate(test_features, test_labels,
                                 batch_size=BATCH_SIZE, verbose=0)
        return results


def save_np_as_csv(data, path, print_flag=True):
    np.savetxt(path, data, delimiter=',')
    if print_flag:
        print(path, 'saved')


if __name__ == '__main__':
    clf = Classifier(DATA_PATH)
    methods = [clf.METHOD_FNN, clf.METHOD_PCA, clf.METHOD_HNN]
    method_names = ["fnn", "pca", "hnn"]
    for i, method in enumerate(methods):
        results = []
        for _ in range(20):
            save_name = method_names[i]+"-"+str(_)
            results.append(clf.run(method, save_name+".png", save_name+".h5"))
        save_np_as_csv(np.array(results), os.path.join(
            OUTPUT_FOLDER, method_names[i]+".csv"))
    print(results)

