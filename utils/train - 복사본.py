import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers.wrappers import TimeDistributed

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config
from utils.dataloader import prepare_data

class MODEL():
    def __init__(self):
        config = get_config()
        self.dimension = config.mfc*3
        self.n_class = config.n_class
        self.n_frame = config.n_frame
        
        self.n_unit = config.n_unit_2
        self.n_unit_1 = config.n_unit_1
        self.n_unit_2 = config.n_unit_2
        self.kernel = config.kernel_size
        self.filters = config.conv_filters
        self.pool = config.pool_size
        self.padding = config.padding
        self.dropout = config.dropout_rate
        
        
    def cnn_lstm(self):
        inputs = keras.Input([self.dimension, config.n_frame, 1, 1])
        conv1 = TimeDistributed(keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding
                                                    , activation=tf.nn.relu))(inputs)
        #nor1 = TimeDistributed(keras.layers.BatchNormalization())(conv1)
        #relu1 = TimeDistributed(keras.layers.ReLU())(nor1)
        pool1 = TimeDistributed(keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding))(conv1)
        conv2 = TimeDistributed(keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding
                                                    , activation=tf.nn.relu))(pool1)
        #nor2 = TimeDistributed(keras.layers.BatchNormalization())(conv2)
        #relu2 = TimeDistributed(keras.layers.ReLU())(nor2)
        pool2 = TimeDistributed(keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding))(conv2)
        drop1 = TimeDistributed(keras.layers.Dropout(rate=self.dropout))(pool2)
        flat = TimeDistributed(keras.layers.Flatten())(drop1)
        
        lstm_1 = keras.layers.LSTM(units=self.n_unit, activation=tf.nn.sigmoid, return_sequences=True)(flat)
        lstm_2 = keras.layers.LSTM(units=self.n_unit, activation=tf.nn.sigmoid)(lstm_1)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(lstm_2)
        
        return tf.keras.Model(inputs=inputs, outputs=output_layer)
    
    def cnn(self):
        inputs = keras.Input([self.dimension, config.n_frame, 1])
        conv1 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding, activation=tf.nn.relu)(inputs)
        nor1 = keras.layers.BatchNormalization()(conv1)
        relu1 = keras.layers.ReLU()(nor1)
        pool1 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(relu1)
        conv2 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding, activation=tf.nn.relu)(pool1)
        nor2 = keras.layers.BatchNormalization()(conv2)
        relu2 = keras.layers.ReLU()(nor2)
        pool2 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(relu2)
        drop1 = keras.layers.Dropout(rate=self.dropout)(pool2)
        flat = keras.layers.Flatten()(drop1)
        hid1 = keras.layers.Dense(self.n_unit_1, activation=tf.nn.relu)(flat)
        drop2 = keras.layers.Dropout(rate=self.dropout)(hid1)
        hid2 = keras.layers.Dense(self.n_unit_2, activation=tf.nn.relu)(drop2)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(hid2)
                
        return tf.keras.Model(inputs=inputs, outputs=output_layer)
    
    def train(self, ctrl, data_path, model_path):
        train_x, train_y = prepare_data(data_path, ctrl)
        valid_x, valid_y = prepare_data(data_path.replace('train', 'eval'), ctrl.replace('train', 'eval'))
        
        train_model1 = self.cnn_lstm()

        train_model1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model1.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint1])

        train_model1.load_weights(model_path)

        # Save the entire model
        train_model.save(model_path)
        
        
        train_model2 = self.cnn()
        
        train_model2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model2.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint2])

        train_model2.load_weights(model_path)

        # Save the entire model
        train_model2.save(model_path)


if __name__ == "__main__":
    config = get_config()

    ctrl = config.ctrl
    data_path = config.feat_path + '/train'
    model_path = config.save_path

    new_model = MODEL()
    new_model.train(ctrl, data_path, model_path)