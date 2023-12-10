import pandas as pd
import numpy as np
import math
import sys

from sklearn.preprocessing import StandardScaler

from pyhht.emd import EMD
# from PyEMD import EMD
from _Support.support_SWT import *
from _Support.support_VMD import VMD
from _Support.support_SSA import SSA
from ewtpy.ewtpy import EWT1D


class dataLoading:

	def __init__(self, file, interval):
		self.file = file
		self.interval = interval
	def data_check(self):
		
		data = pd.read_csv(self.file, encoding='gbk').values
		# data = pd.read_csv(self.file)
		# data = data["Power consumption"].values
		data_checked = data.T.tolist()
		# print(data_checked)
		# for i in range(len(data_checked)):
		for i in range(len(data_checked)):
			data_feature = data_checked[i]
			for j in range(len(data_feature)):
				NaN_judge = math.isnan(data_feature[j])
				if NaN_judge is True or data_feature[j] == 0:
					if j == 0:  # mean of first.
						h1 = 1
						while math.isnan(data_checked[i][j + h1]) is True:
							h1 = h1 + 1
						mean = data_checked[i][j + h1]
					elif j == len(data) - 1:  # mean of last.
						h2 = - 1
						while math.isnan(data_checked[i][j + h2]) is True:
							h2 = h2 - 1
						mean = data_checked[i][j + h2]
					else:
						h1 = 1
						while math.isnan(data_checked[i][j + h1]) is True:
							h1 = h1 + 1
						h2 = - 1
						while math.isnan(data_checked[i][j + h2]) is True:
							h2 = h2 - 1
						mean = (data_checked[i][j + h1] + data_checked[i][j + h2]) / 2
					data_feature[j] = mean
			data_checked[i] = data_feature

		data_checked = np.array(data_checked).reshape(-1,).T.tolist()

		data_itved = []
		data_itved_length = len(data_checked) // self.interval
		
		for i in range(data_itved_length):
			# data_itved.append(round(sum(data_checked[i * self.interval:(i + 1) * self.interval]), 8))
			data_itved.append(round(sum(data_checked[i * self.interval:(i + 1) * self.interval]), 8))
		return np.array(data_itved)
	
	def EXECUTE(self):
		data = self.data_check()
		return data
	

class dataProcessing:
	
	def __init__(self, file, order, interval, start, train_set, test_set, step):
		
		self.name_list = ['Cement Factory', 'Forge Factory', 'Metal Factory', 'Paper Factory', 'Steel Factory']
		self.print_list = ['Cement Factory', 'Forge Factory', 'Metal Factory', 'Paper Factory', 'Steel Factory']
		
		self.file = file
		self.order = order
		self.interval = interval
		
		self.start_num = start
		self.train_num = train_set
		self.test_num = test_set
		self.step_num = step
		self.total_num = self.start_num + self.train_num + self.test_num + self.step_num
		
		self.data = dataLoading(self.file, interval=self.interval).EXECUTE()
		
		print(self.name_list[self.order] + ' Data with Interval of ' + str(self.interval * 5) + ' minutes Loaded Successfully.')
		
	def __create_data(self, data):
		TS_X = []
		
		for i in range(data.shape[0] - self.step_num):
			b = data[i:(i + self.step_num), 0]
			TS_X.append(b)    #(9600,8)

		dataX1 = TS_X[:self.train_num]     #(8400,8)
		dataX2 = TS_X[self.train_num:]     #(1200,8)
		dataY1 = data[self.step_num: self.train_num + self.step_num, 0]  #(8400,)
		dataY2 = data[self.train_num + self.step_num:, 0]  #(1200,)
		
		return [np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)]
	
	def __coeffs_to_list(self, coeffs):
		level_list = []
		for level in range(len(coeffs)):
			trainX, trainY, testX, testY = self.__create_data(coeffs[level])
			level_part = [trainX, trainY, testX, testY]
			level_list.append(level_part)
		return level_list
	
	def __preprocessing_regular(self, data):
		data = self.__create_data(data)
		return data
	
	def __preprocessing_EMD(self, data):
		decomposer = EMD(data)
		imfs = decomposer.decompose()
		result_emd = imfs.tolist()
		for i in range(len(result_emd)):
			result_emd[i] = np.array(result_emd[i]).reshape(-1, 1)

		result_emd = self.__coeffs_to_list(result_emd)

		return result_emd


	def __preprocessing_SWT(self, data):
		# wavefun = pywt.Wavelet('db1')
		wavefun = pywt.Wavelet('bior1.5')
		result_SWT = swt_decom(data, wavefun, 3)
		
		result_SWT = self.__coeffs_to_list(result_SWT)
		return result_SWT
	
	def __preprocessing_EWT(self, data):
		result_ewt, _, _ = EWT1D(data.reshape(-1, ), N=4)
		result_ewt = result_ewt.T.tolist()
		for i in range(len(result_ewt)):
			result_ewt[i] = np.array(result_ewt[i]).reshape(-1, 1)
		
		result_ewt = self.__coeffs_to_list(result_ewt)
		return result_ewt
	
	def __preprocessing_VMD(self, data):
		VMD_level = 4
		imf_list = VMD(data.reshape(-1, ), VMD_level)
		imf_list = imf_list.tolist()
		
		result_VMD = []
		for i in range(len(imf_list)):
			imf = imf_list[i]
			for j in range(len(imf)):
				part_real = imf[j].real
				imf[j] = part_real
			result_VMD.append(np.array(imf).reshape(-1, 1))
		
		coeffs_rest = 0
		for i in range(len(result_VMD)):
			coeffs_rest = coeffs_rest + result_VMD[i]
		coeffs_rest = data - coeffs_rest
		
		result_VMD[0] = result_VMD[0] + result_VMD[1]
		result_VMD[1] = coeffs_rest
		
		result_VMD = self.__coeffs_to_list(result_VMD)
		return result_VMD
	
	def __preprocessing_SSA(self, data):
		result_ssa = SSA(data.reshape(-1, ), 4)
		result_ssa = result_ssa.tolist()
		for i in range(len(result_ssa)):
			result_ssa[i] = np.array(result_ssa[i]).reshape(-1, 1)
		result_ssa = self.__coeffs_to_list(result_ssa)

		return result_ssa
	
	def __deal_flag(self, data):
		
		flag_temp = []
		
		for i in range(len(data) - 1):
			if (data[i + 1] - data[i]) > 0:
				flag_temp.append(1)
			elif (data[i + 1] - data[i]) < 0:
				flag_temp.append(2)
			else:
				flag_temp.append(0)
		
		return flag_temp
	
	def size_judge(self):
		if self.data.shape[0] < self.total_num:
			print('This Dataset is not enough for such trainNum & testNum.')
			sys.exit(1)
	
	def data_preprocessing(self):
		
		dict_save = {}
	
		targetData = self.data
		
		targetData = targetData[self.start_num + 1: self.start_num + self.train_num + self.test_num + 1]
		targetData = targetData.reshape(-1, 1)
		scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
		dict_save['scaler_target'] = scaler_target
		
		targetData = scaler_target.fit_transform(targetData)

		data_regular = self.__preprocessing_regular(targetData)
		dict_save['data_regular'] = data_regular
		data_EMD = self.__preprocessing_EMD(targetData)
		dict_save['data_EMD'] = data_EMD
		data_SWT = self.__preprocessing_SWT(targetData)
		dict_save['data_SWT'] = data_SWT
		data_EWT = self.__preprocessing_EWT(targetData)
		dict_save['data_EWT'] = data_EWT
		data_VMD = self.__preprocessing_VMD(targetData)
		dict_save['data_VMD'] = data_VMD
		data_SSA = self.__preprocessing_SSA(targetData)
		dict_save['data_SSA'] = data_SSA
		
		y_real = scaler_target.inverse_transform(data_regular[3])
		y_flag = self.__deal_flag(y_real)
		dict_save['y_real'] = y_real
		dict_save['y_flag'] = y_flag
		
		dict_save['name'] = self.name_list[self.order]
		dict_save['print'] = self.print_list[self.order]
		
		np.save('saved\\Dataset'+str(self.order+1)+'_interval'+str(self.interval * 5)+'min_ts'+str(self.step_num)+'.npy', dict_save)
		
		print(self.name_list[self.order] + ' Data with Interval of ' + str(self.interval * 5) + ' minutes Preprocessed and Saved.\n')
	
	def EXECUTE(self):
		self.size_judge()
		self.data_preprocessing()
		
	
def Data(order, interval, startNum, trainNum, testNum, aheadNum):
	
	# file = 'dataset\\house' + str(order + 1) + '_5min_KWh.csv'
	file= 'datasets\\Dataset' + str(order + 1) + '.csv'

	trainNum = trainNum // interval
	print("TrainNum:"+str(trainNum))

	testNum = (testNum // interval) + aheadNum
	print("TestNum:"+str(testNum))

	dataProcessing(file, order, interval, startNum, trainNum, testNum, aheadNum).EXECUTE()
