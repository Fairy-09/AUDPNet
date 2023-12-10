import time
from keras.callbacks import Callback
from keras.models import Input, Model, Sequential                     #### TCN
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional, Flatten, Add, Concatenate, MaxPool1D, LeakyReLU, GRU
import keras.backend as K

from _Support.support_NLSTM import NestedLSTM


class TimeHistory(Callback):
	def on_train_begin(self, logs={}):
		self.times = []
		self.totaltime = time.time()
	
	def on_train_end(self, logs={}):
		self.totaltime = time.time() - self.totaltime
	
	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()
	
	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)


def build_GRU(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = GRU(64)(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model


def build_LSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	# x = LSTM(64)(i)
	x = LSTM(64)(i)
	x = Dense(16, activation='linear')(x)
	# o = Dense(1, activation="linear")(x)
	o = Dense(1, activation="softmax")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model


def build_SLSTM(timestep):
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = Reshape((-1, 1))(i)
	x = LSTM(64, return_sequences=True)(x)
	x = LSTM(64, return_sequences=True)(x)
	x = LSTM(64)(x)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	return model


def build_BiLSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = Bidirectional(LSTM(64), merge_mode='concat')(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	
	model.compile(optimizer='Adam', loss='mse', )
	model.summary()
	
	return model


def build_NLSTM(timestep):
	
	batch_size, timesteps, input_dim = None, timestep, 1
	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	
	x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)
	x = Dense(16, activation='linear')(x)
	o = Dense(1, activation="linear")(x)
	
	model = Model(inputs=[i], outputs=[o])
	model.compile(optimizer='rmsprop', loss='mse', )
	model.summary()
	
	return model

def build_MC_BiNLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1

    input_A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x0 = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(input_A3)
    x0 = Dense(16, activation='linear')(x0)
    model0 = Model(inputs=input_A3, outputs=x0)

    input_D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x1 = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(input_D1)
    x1 = Dense(16, activation='linear')(x1)
    model1 = Model(inputs=input_D1, outputs=x1)

    input_D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x2 = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(input_D2)
    x2 = Dense(16, activation='linear')(x2)
    model2 = Model(inputs=input_D2, outputs=x2)

    input_D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x3 = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(input_D3)
    x3 = Dense(16, activation='linear')(x3)
    model3 = Model(inputs=input_D3, outputs=x3)

    # o = Add()([model0.output, model1.output, model2.output, model3.output])

    combined = Concatenate(axis=1)([model0.output, model1.output, model2.output, model3.output])
    o = Dense(16, activation="linear")(combined)

    o = Dense(1, activation="linear")(o)

    model = Model(inputs=[model0.input, model1.input, model2.input, model3.input], outputs=o)

    model.compile(optimizer='rmsprop', loss='mse')
    model.summary()

    return model