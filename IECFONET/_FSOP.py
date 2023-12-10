import numpy as np

from sklearn import tree
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor
from sklearn import svm

from _Support.support_SWT import *
from _Model.model_major import *
from _Part.part_evaluate import MAE1, RMSE1, MAPE1, R2


class mainFunc_FORECASTING:
	def __init__(self, step_num, order, interval):
		
		# self.dict_load = np.load('saved\\dataset_house'+str(order)+'_interval'+str(interval * 5)+'min_ts'+str(step_num)+'.npy',
		#                         allow_pickle=True).item()
		self.dict_load = np.load('saved\\Dataset'+str(order+1)+'_interval'+str(interval * 5)+'min_ts'+str(step_num)+'.npy',
		                        allow_pickle=True).item()
		self.data_regular = self.dict_load['data_regular']
		self.data_EMD = self.dict_load['data_EMD']
		self.data_SWT = self.dict_load['data_SWT']
		self.data_EWT = self.dict_load['data_EWT']
		self.data_VMD = self.dict_load['data_VMD']
		self.data_SSA = self.dict_load['data_SSA']
		self.y_real = self.dict_load['y_real']
		self.y_flag = self.dict_load['y_flag']
		self.scaler_target = self.dict_load['scaler_target']
		
		self.name = self.dict_load['name']
		self.print_name = self.dict_load['print']
		
		self.step_num = step_num
		self.order = order
		self.interval = interval
		
		self.result_list = []
		self.result_print = '\nEvaluation.'
		self.result_save = []
		self.dict_save = {}

		print(self.name + ' Dataset is Ready.\n')
	
	###############################################################################
	
	def __TrainPredict_ML(self, model, data):
		x_train = data[0]
		y_train = data[1]
		x_test = data[2]

		time_start = time.time()
		model.fit(x_train, y_train)
		time_end = time.time()
		train_time = time_end - time_start

		predict = model.predict(x_test).reshape(-1, )
		return predict, train_time
	
	def __TrainPredict_DL(self, model, data):
		x_train = data[0]
		y_train = data[1]
		x_test = data[2]
		
		x_train = x_train.reshape(-1, self.step_num, 1)
		x_test = x_test.reshape(-1, self.step_num, 1)

		time_callback = TimeHistory()
		history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=1,
		                    shuffle=False, callbacks=[time_callback])
		print("history.history:{}".format(history.history))
		# plt.plot(history.history['loss'])
		# plt.plot(history.history['val_loss'])
		# plt.title('model train vs validation loss')
		# plt.ylabel('loss')
		# plt.xlabel('epoch')
		# plt.legend(['train', 'validation'], loc='upper right')
		# plt.show()

		train_time = time_callback.totaltime
		
		predict = model.predict(x_test).reshape(-1, )
		return predict, train_time
		
	###############################################################################
	
	def __deal_flag_accuracy(self, predict):

		predict_flag = []

		for i_flag in range(len(predict) - 1):
			if (predict[i_flag + 1] - predict[i_flag]) > 0:
				predict_flag.append(1)
			elif (predict[i_flag + 1] - predict[i_flag]) < 0:
				predict_flag.append(2)
			else:
				predict_flag.append(0)

		rightCount = 0
		for j_flag in range(len(predict_flag)):
			if self.y_flag[j_flag] == predict_flag[j_flag]:
				rightCount = rightCount + 1
		accuracy = rightCount / len(predict_flag)

		return accuracy * 100

	def __evaluate(self, name, name_short, predict, train_time):
		
		mae = MAE1(self.y_real, predict)
		rmse = RMSE1(self.y_real, predict)
		mape = MAPE1(self.y_real, predict)
		r2 = R2(self.y_real, predict)
		
		accuracy = self.__deal_flag_accuracy(predict)
		
		result_print = '\n\nMAE_' + name + ': {}'.format(mae)
		result_print += '\nRMSE_' + name + ': {}'.format(rmse)
		result_print += '\nMAPE_' + name + ': {}'.format(mape)
		result_print += '\nR2_' + name + ': {}'.format(r2)
		result_print += '\nACC_' + name + ': {}'.format(accuracy)
		result_print += '\nTIME_' + name + ': {}'.format(train_time)
		result_csv = [name_short, mae, rmse, mape, r2, accuracy, train_time]
		
		self.result_print += result_print
		self.result_save.append(result_csv)

	def __postpredict(self, predict, name, name_short, train_time):
		predict = self.scaler_target.inverse_transform(predict)
		self.__evaluate(name, name_short, predict, train_time)
		self.result_list.append([[name, name_short], predict])
		# self.dict_save['predict_'+str(name)] = predict
		
	###############################################################################
		
	def FSOP_ML(self, model, name, name_short, data):
		
		print(name + ' Start.')
		
		predict, train_time = self.__TrainPredict_ML(model, data)
		self.__postpredict(predict, name, name_short, train_time)
		
		print(name + ' Complete.\n')
		
	def FSOP_DL(self, model, name, name_short, data):
		
		print(name + ' Start.')

		predict, train_time = self.__TrainPredict_DL(model, data)
		self.__postpredict(predict, name, name_short, train_time)
		
		K.clear_session()
		del model
		print(name + ' Complete.\n')

	def FSOP_DL_loop(self, model, name, name_short, data):
		
		print(name + ' Start.')
		
		train_time_total = 0
		predict_list = []
		for subSignal_index in range(len(data)):
			
			print(str(name) + '_level ' + str(subSignal_index + 1) + ' Start.')

			predict, train_time = self.__TrainPredict_DL(model, data[subSignal_index])
			
			train_time_total = train_time_total + train_time
			predict_list.append(predict)
			print(str(name) + '_level ' + str(subSignal_index + 1) + ' Complete.\n')
		
		predict_list = np.array(predict_list).T.tolist()
		for j in range(len(predict_list)):
			predict_list[j] = sum(predict_list[j])
		predict = np.array(predict_list).reshape(-1, )
		
		self.__postpredict(predict, name, name_short, train_time_total)
		
		K.clear_session()
		del model
		print(name + ' Complete.\n')
	
	def FSOP_DL_loop_SWT(self, model, name, name_short, data):
		
		print(name + ' Start.')
		
		train_time_total = 0
		predict_list = []
		for subSignal_index in range(len(data)):
			print(str(name) + '_level ' + str(subSignal_index + 1) + ' Start.')
			
			predict, train_time = self.__TrainPredict_DL(model, data[subSignal_index])
			
			train_time_total = train_time_total + train_time
			predict_list.append(predict)
			print(str(name) + '_level ' + str(subSignal_index + 1) + ' Complete.\n')
		
		wavefun = pywt.Wavelet('db1')
		predict = iswt_decom(predict_list, wavefun)
		
		self.__postpredict(predict, name, name_short, train_time_total)
		
		K.clear_session()
		del model
		print(name + ' Complete.\n')
	
	###############################################################################
	
	def Decision_Tree(self):
		return tree.DecisionTreeRegressor()
	
	def Random_Forest(self):
		return ensemble.RandomForestRegressor(n_estimators=50)
	
	def SVR(self):
		return svm.SVR()
	
	def MLP(self):
		return MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)
	
	def LSTM(self):
		return build_LSTM(self.step_num)
	
	def GRU(self):
		return build_GRU(self.step_num)
	
	def NLSTM(self):
		return build_NLSTM(self.step_num)
	
	def BiLSTM(self):
		return build_BiLSTM(self.step_num)
	
	def SLSTM(self):
		return build_SLSTM(self.step_num)

	# def MC_BiNLSTM(self):
	# 	return build_MC_BiNLSTM(self.step_num)

	###############################################################################
	
	def EXECUTE(self):
		

		self.FSOP_ML(self.Random_Forest(), 'RandomForest', 'RF', self.data_regular)
		self.FSOP_ML(self.SVR(), 'SVR', 'SVR', self.data_regular)


		self.FSOP_DL(self.GRU(), 'GRU', 'GRU', self.data_regular)

		self.FSOP_DL_loop(self.SLSTM(), 'EMD-SLSTM', 'EMD-SLSTM', self.data_EMD)
		self.FSOP_DL_loop(self.BiLSTM(), 'EMD-BI-LSTM', 'EMD-BI-LSTM', self.data_EMD)



		self.FSOP_DL_loop(self.LSTM(), 'EWT-LSTM', 'EWT-LSTM', self.data_EWT)


		self.FSOP_DL_loop(self.LSTM(), 'VMD-LSTM', 'VMD-LSTM', self.data_VMD)

		self.FSOP_DL_loop(self.LSTM(), 'SSA-PLSTM', 'SSA-PLSTM', self.data_SSA)
		self.FSOP_DL_loop_SWT(self.BiLSTM(), 'SWTBL', 'SWTBL', self.data_SWT)


		self.dict_save['result_list'] = self.result_list
		self.dict_save['result_print'] = self.result_print
		self.dict_save['result_save'] = self.result_save
		self.dict_save['y_real'] = self.y_real
		
		self.dict_save['name'] = self.name
		self.dict_save['print'] = self.print_name
		
		np.save('saved\\predict_Dataset'+str(self.order+1)+'_interval'+str(self.interval * 5)+'min_ts'+str(self.step_num)+'.npy', self.dict_save)
		
		print(self.name + 'Data with Interval of ' + str(self.interval * 5) + ' minutes Train and Predict Complete.\n')
		return self.dict_save


def FSOP(ahead_num, order, interval):
	mainFunc_FORECASTING(ahead_num, order, interval).EXECUTE()
