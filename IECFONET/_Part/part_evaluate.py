
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def MAPE1(true,predict):

    L1 = int(len(true))
    L2 = int(len(predict))

    if L1 == L2:

        SUM = 0.0
        for i in range(L1):
            if true[i] == 0:
                SUM = abs(predict[i]) + SUM
            else:
                SUM = abs((true[i] - predict[i]) / true[i]) + SUM
        per_SUM = SUM * 100.0
        mape = per_SUM / L1
        return mape
    else:
        print("error")


def RMSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    rmse = math.sqrt( mean_squared_error(testY[:], testPredict[:]))
    return rmse


def MAE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mae=mean_absolute_error(testY[:], testPredict[:])
    return mae


def R2(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score











    