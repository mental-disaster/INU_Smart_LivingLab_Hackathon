import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers, initializers, layers
from keras.models import Model
from keras.regularizers import l2
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
        
        self.n_unit_1 = config.n_unit_1
        self.n_unit_2 = config.n_unit_2
        self.n_unit_3 = config.n_unit_3
        self.kernel = config.kernel_size
        self.filters = config.conv_filters
        self.pool = config.pool_size
        self.padding = config.padding
        self.dropout = config.dropout_rate
    
    def cnn(self):
        inputs = keras.Input([self.dimension, config.n_frame, 1])
        conv1 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding)(inputs)
        nor1 = keras.layers.BatchNormalization()(conv1)
        elu1 = keras.layers.ELU()(nor1)
        pool1 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(elu1)
        conv2 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding)(pool1)
        nor2 = keras.layers.BatchNormalization()(conv2)
        elu2 = keras.layers.ELU()(nor2)
        pool2 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(elu2)
        conv3 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding)(pool2)
        nor3 = keras.layers.BatchNormalization()(conv3)
        elu3 = keras.layers.ELU()(nor3)
        pool3 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(elu3)
        drop1 = keras.layers.Dropout(rate=0.4)(pool3)
        flat = keras.layers.Flatten()(drop1)
        hid1 = keras.layers.Dense(self.n_unit_1, activation=tf.nn.elu)(flat)
        drop2 = keras.layers.Dropout(rate=0.4)(hid1)
        hid2 = keras.layers.Dense(self.n_unit_2, activation=tf.nn.elu)(drop2)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(hid2)
        return tf.keras.Model(inputs=inputs, outputs=output_layer)
    
    def lstm(self):
        inputs = keras.Input([self.dimension, config.n_frame])
        lstm_1 = keras.layers.LSTM(units=self.n_unit_1, dropout=0.3, return_sequences=True)(inputs)
        lstm_2 = keras.layers.LSTM(units=self.n_unit_1, dropout=0.5)(lstm_1)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(lstm_2)
        return keras.Model(inputs=inputs, outputs=output_layer)
    
    def cnn_lstm(self):
        #cnn
        inputs = keras.Input([self.dimension, config.n_frame, 1])
        conv1 = TimeDistributed(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel, padding=self.padding))(inputs)
        nor1 = TimeDistributed(keras.layers.BatchNormalization())(conv1)
        elu1 = TimeDistributed(keras.layers.ELU())(nor1)
        pool1 = TimeDistributed(keras.layers.MaxPool1D(pool_size=self.pool, padding=self.padding))(elu1)
        conv2 = TimeDistributed(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel, padding=self.padding))(pool1)
        nor2 = TimeDistributed(keras.layers.BatchNormalization())(conv2)
        elu2 = TimeDistributed(keras.layers.ELU())(nor2)
        pool2 = TimeDistributed(keras.layers.MaxPool1D(pool_size=self.pool, padding=self.padding))(elu2)
        drop1 = TimeDistributed(keras.layers.Dropout(rate=0.3))(pool2)
        flat = TimeDistributed(keras.layers.Flatten())(drop1)
        #lstm
        lstm_1 = keras.layers.LSTM(units=self.n_unit_1, dropout=0.4, return_sequences=True)(flat)
        lstm_2 = keras.layers.LSTM(units=self.n_unit_1, dropout=0.4)(lstm_1)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(lstm_2)
        return tf.keras.Model(inputs=inputs, outputs=output_layer)
    
    def cnn2(self):
        inputs = keras.Input([self.dimension, config.n_frame, 1])
        conv1 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel,
                                    padding=self.padding, activation=tf.nn.relu)(inputs)
        pool1 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(conv1)
        conv2 = keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel, padding=self.padding, activation=tf.nn.relu)(pool1)
        pool2 = keras.layers.MaxPool2D(pool_size=self.pool, padding=self.padding)(conv2)
        drop1 = keras.layers.Dropout(rate=0.3)(pool2)
        flat = keras.layers.Flatten()(drop1)
        hid1 = keras.layers.Dense(self.n_unit_1, activation=tf.nn.relu)(flat)
        drop2 = keras.layers.Dropout(rate=0.3)(hid1)
        hid2 = keras.layers.Dense(self.n_unit_2, activation=tf.nn.relu)(drop2)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(hid2)
        return tf.keras.Model(inputs=inputs, outputs=output_layer)
    
    def ensemble(self, models):
        inputs = keras.Input([self.dimension, config.n_frame, 1])
        outputs = []
        for i in models:
            outputs.append(i(inputs))
        y = layers.Average()(outputs)
        return Model(inputs = inputs, outputs = y, name='ensemble')
    
    def train(self, ctrl, data_path, model_path):
        train_x, train_y = prepare_data(data_path, ctrl)
        valid_x, valid_y = prepare_data(data_path.replace('train', 'eval'), ctrl.replace('train', 'eval'))
        
        
        train_model1 = self.cnn()
        
        
        train_model1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        
        
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_1.h5', verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model1.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint1])

        train_model1.load_weights('./model/model_1.h5')

        # Save the entire model
        train_model1.save('./model/model_1.h5')
        
                
        
        train_model2 = self.lstm()
        
        '''
        train_model2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        
        
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_2.h5', verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model2.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint2])

        train_model2.load_weights('./model/model_2.h5')

        # Save the entire model
        train_model2.save('./model/model_2.h5')
        '''
        
        '''
        train_model3 = self.cnn_lstm()
        '''
        '''
        train_model3.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        
        
        checkpoint3 = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_3.h5', verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model3.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint3])

        train_model3.load_weights('./model/model_3.h5')

        # Save the entire model
        train_model3.save('./model/model_3.h5')
        '''
        
        '''
        train_model4 = self.cnn2()
        
        
        train_model4.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        
        
        checkpoint4 = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_4.h5', verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model4.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint4])

        train_model4.load_weights('./model/model_4.h5')

        # Save the entire model
        train_model4.save('./model/model_4.h5')
        '''
        '''
        train_model5 = self.lstm2()
        
        
        train_model5.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
        
        
        checkpoint5 = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_5.h5', verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model5.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint5])

        train_model5.load_weights('./model/model_5.h5')

        # Save the entire model
        train_model5.save('./model/model_5.h5')
        '''
        
        '''
        train_model1 = tf.keras.models.load_model('./model/model_1.h5')
        train_model2 = tf.keras.models.load_model('./model/model_2.h5')
        train_model3 = tf.keras.models.load_model('./model/model_3.h5')
        train_model4 = tf.keras.models.load_model('./model/model_4.h5')
        '''
        '''
        models = [train_model1, train_model2, train_model4]

        train_model = self.ensemble(models)

        train_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=0, monitor='val_accuracy',
                                                        save_best_only=True)
        train_model.fit(train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_data=(valid_x, valid_y),
              callbacks=[checkpoint])
        train_model.load_weights(model_path)

        train_model.save(model_path)
        '''
        

if __name__ == "__main__":
    config = get_config()

    ctrl = config.ctrl
    data_path = config.feat_path + '/train'
    model_path = config.save_path

    new_model = MODEL()
    new_model.train(ctrl, data_path, model_path)