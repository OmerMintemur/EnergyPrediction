# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:47:13 2022

@author: OMER
"""
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt


# ---------------- READ THE DATA ----------------
data = pd.read_csv('D:\\\Projeler\\Energy_Prediction\\Codes\\household_power_consumption.txt', sep=";")


# ---------------- PRINT THE DATA HEAD ---------------
print(data.head())


# ---------------- DROP THE NAN VALUES ---------------
print("Before Dropping Nan Values")

print(data.isnull().values.any())
print(data.isnull().sum().sum())

data = data.dropna()


# ---------------- GET TYPES OF EACH COLUMN ---------------
# print(data.dtypes)


# ---------------- CHANGE EACH COLUMN DTYPES INTO ITS PROPER VALUES THEN SHOW ---------------
data['Time'] = pd.to_datetime(data['Time'])
data['Time_Hour']= data['Time'].dt.hour
data['Time_Minute']= data['Time'].dt.minute
data['Time_Second']= data['Time'].dt.second
data['Date'] = pd.to_datetime(data['Date'])
# data['Date'] = pd.to_datetime(data['Date']).dt.date
data = data.astype({'Global_active_power':'float',
                    'Global_reactive_power':'float',
                    'Voltage':'float',
                    'Global_intensity':'float',
                    'Sub_metering_1':'float',
                    'Sub_metering_2':'float',
                    'Sub_metering_3':'float'})
# print(data.dtypes)

# print(data.head())


# ------------------- DROP THE Time COLUMN ----------------------------
data.drop('Time', axis=1, inplace=True)
data.drop('Time_Hour', axis=1, inplace=True)
data.drop('Time_Minute', axis=1, inplace=True)
data.drop('Time_Second', axis=1, inplace=True)




# ------------------ AGGREGATE INTO ONE DAY ---------------------
data_resample = data.resample('D', on='Date').sum()

# ----------------- CHANGE TIME SERIES INTO SUPERVISED LEARNING PROBLEM -----------------

def series_to_supervised(data,previous_steps=1, forecast_steps=1, dropnan=True):
      
      col_names = data.columns
      cols, names = list(), list()
      for i in range(previous_steps, 0, -1):
            cols.append(data.shift(i))
            names +=[('%s(t-%d)' % (col_name, i)) for col_name in col_names]
            
      for i in range(0,forecast_steps):
            cols.append(data.shift(-i))
            if i == 0:
                  names +=[('%s(t)' % col_name) for col_name in col_names]
            else:
                  names += [('%s(t+%d)' % (col_name,i)) for col_name in col_names]
      
      agg = pd.concat(cols, axis = 1)
      agg.columns = names
      
      
      if dropnan:
            agg.dropna(inplace=True)
            
      return agg


      
data_one_day_prediction = series_to_supervised(data_resample,1,1,True) 
data_three_days_prediction = series_to_supervised(data_resample,3,3,True)
data_five_days_prediction = series_to_supervised(data_resample,5,5,True)   

      


def split_data_into_train_test_validation(dataframe, test_percentage = 0.8, train_percentage = 0.1, valid_percentage = 0.1):
      train = int(len(data_one_day_prediction)*0.8)
      test = int(len(data_one_day_prediction)*0.2)
      # valid = int(len(data_one_day_prediction)*0.1)

      train_data = data_one_day_prediction.iloc[0:train]
      test_data = data_one_day_prediction.iloc[train:train+test]
      # valid_data = data_one_day_prediction.iloc[train+test:train+test+valid]
      
      return train_data, test_data

def return_X_Y_single(train_data,test_data):
      train_data_x = train_data.loc[:,train_data.columns!='Global_active_power(t)']
      train_data_y = train_data['Global_active_power(t)']
      
      test_data_x = test_data.loc[:,test_data.columns!='Global_active_power(t)']
      test_data_y = test_data['Global_active_power(t)']
      
      # valid_data_x = valid_data.loc[:,valid_data.columns!='Global_active_power(t)']
      # valid_data_y = valid_data['Global_active_power(t)']
      
      return train_data_x,train_data_y, test_data_x, test_data_y
      
#------------------------MAKE PREDICTION FOR ONE DAY (GLOBAL VOLTAGE POWER)-----------------------

#DIVIDE %80 PERCENT FOR TRAINING %10 PERCENT FOR TEST AND %10 PERCENT FOR VALIDATION
train_data, test_data= split_data_into_train_test_validation(data_one_day_prediction)

#RETURN train_data_x, train_data_y, test_data_x, test_data_y
#We Try to Predict GLOBAL_ACTIVE_POWER

train_data_x, train_data_y, test_data_x,test_data_y = return_X_Y_single(train_data,test_data)

print(train_data.index)
beingsaved = plt.figure()
plt.plot(train_data.index, train_data['Global_active_power(t)'],color='black',label="Test Data")
plt.plot(test_data.index, test_data['Global_active_power(t)'],color='red', label="Train Data")
plt.grid()
plt.title('Dataset Division')
plt.ylabel("Global Active Power (KW)")
plt.xlabel("Dates")
plt.axvline(x=test_data.index[0], ymax=0.8, color='blue', linestyle='--',label="Division Point")
plt.text(test_data.index[0], 4500, str((test_data.index[0].strftime('%m-%d-%Y'))), color='Blue',ha='center', va='top')
# plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
plt.legend()
plt.tight_layout()
beingsaved.autofmt_xdate()
plt.show()
beingsaved.savefig('Dataset Division.png', format='png', dpi=1200, bbox_inches='tight')





#Import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error

# We need to scale our data since values are so diversed
from sklearn.preprocessing import MinMaxScaler

normalizer = MinMaxScaler()
train_data_x = normalizer.fit_transform(train_data_x)
test_data_x = normalizer.transform(test_data_x)


# These codes for number of estimators------------------------------------
rf_mses = []
et_mses = []
number_of_est_dict_et={}
number_of_est_dict_rf={}
x_mses = np.arange(1,150,1)

for x in range(1,150,1):

      et_regression = ExtraTreesRegressor(n_estimators=x, random_state=0).fit(train_data_x, train_data_y)
      rf_regression = RandomForestRegressor(n_estimators=x, random_state=0).fit(train_data_x, train_data_y)
      
      prediction_results_et = et_regression.predict(test_data_x)
      prediction_results_rf = rf_regression.predict(test_data_x) 
      
      et_mses.append(mean_squared_error(np.array(test_data_y), prediction_results_et,squared=True))
      rf_mses.append(mean_squared_error(np.array(test_data_y), prediction_results_rf,squared=True))

      number_of_est_dict_et[x]=mean_squared_error(np.array(test_data_y), prediction_results_et)
      number_of_est_dict_rf[x]=mean_squared_error(np.array(test_data_y), prediction_results_rf)


#Draw the results
print("Smallest error according to Number of estimators(RandomForest), ",min(rf_mses))
print("Smallest error according to Number of estimators(ExtraTree), ",min(et_mses))

print(et_mses)
#Best prediction for ET
lowest = min(number_of_est_dict_et.items(), key=lambda x: x[1])
print(lowest)
et_regression = ExtraTreesRegressor(n_estimators=lowest[0], random_state=0).fit(train_data_x, train_data_y)
predictions=et_regression.predict(test_data_x)
beingsaved = plt.figure()
x_ticks_for_pred = np.arange(0,len(predictions),1)

plt.scatter(test_data.index,predictions,marker="x",c="blue",label="Predictions")
plt.scatter(test_data.index,test_data_y,marker="d",alpha=0.5,c="red", label="True Test Values")
plt.grid()
plt.title('Prediction Graph \n For Extra Tree According to Number of Estimator \n(According the Information Gained 1-Day Before)')
plt.ylabel("Global Active Power (KWH)")
plt.xlabel("Dates")
plt.figtext(0.99, 0.01, 'Model w.r.t Smallest RMSE Value', horizontalalignment='right')
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
beingsaved.autofmt_xdate()
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//Predictions_Est_ET_Number_of_Estimators.png', format='png', dpi=600, bbox_inches='tight')



#Best prediction for RF
lowest = min(number_of_est_dict_rf.items(), key=lambda x: x[1])
print(lowest)
rf_regression = RandomForestRegressor(n_estimators=lowest[0], random_state=0).fit(train_data_x, train_data_y)
predictions=rf_regression.predict(test_data_x)
beingsaved = plt.figure()
x_ticks_for_pred = np.arange(0,len(predictions),1)
plt.scatter(test_data.index,predictions,marker="x",c="blue",label="Predictions")
plt.scatter(test_data.index,test_data_y,marker="d",alpha=0.5,c="red", label="True Test Values")
plt.grid()
plt.title('Prediction Graph\nFor Random Forest According to Number of Estimator\n(According the Information Gained 1-Day Before)')
plt.ylabel("Global Active Power (KWH)")
plt.xlabel("Dates")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
beingsaved.autofmt_xdate()
plt.figtext(0.99, 0.01, 'Model w.r.t Smallest RMSE Value', horizontalalignment='right')
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//Predictions_Est_RF_Number_of_Estimators.png', format='png', dpi=1200, bbox_inches='tight')


# General Results
beingsaved = plt.figure()
plt.plot(x_mses, et_mses,label='ExtraTreeRegression')
plt.plot(x_mses, rf_mses,label='RandomForestRegression')
plt.ylabel('RMSE')
plt.xlabel('Number of Estimator')
plt.title('RMSE According to Estimator Number')
plt.grid()
plt.legend()
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//RMSE_Est.png', format='png', dpi=1200, bbox_inches='tight')
# These codes for number of estimators------------------------------------


# These codes for number of depths
rf_mses = []
et_mses = []
number_of_est_dict_et={}
number_of_est_dict_rf={}
x_mses = np.arange(1,150,1)

for x in range(1,150,1):

      et_regression = ExtraTreesRegressor(max_depth=x, random_state=0).fit(train_data_x, train_data_y)
      rf_regression = RandomForestRegressor(max_depth=x, random_state=0).fit(train_data_x, train_data_y)
      
      prediction_results_et = et_regression.predict(test_data_x)
      prediction_results_rf = rf_regression.predict(test_data_x) 
      et_mses.append(mean_squared_error(np.array(test_data_y), prediction_results_et,squared=True))
      rf_mses.append(mean_squared_error(np.array(test_data_y), prediction_results_rf,squared=True))
      
      number_of_est_dict_et[x]=mean_squared_error(np.array(test_data_y), prediction_results_et)
      number_of_est_dict_rf[x]=mean_squared_error(np.array(test_data_y), prediction_results_rf)
      


#Draw the results
print("Smallest error according to Number of max_depth(RandomForest), ", min(rf_mses))
print("Smallest error according to Number of max_depth(ExtraTree), ", min(et_mses))
import matplotlib.pyplot as plt

#Best prediction for ET
lowest = min(number_of_est_dict_et.items(), key=lambda x: x[1])
print(lowest)
et_regression = ExtraTreesRegressor(n_estimators=lowest[0], random_state=0).fit(train_data_x, train_data_y)
predictions=et_regression.predict(test_data_x)
beingsaved = plt.figure()
x_ticks_for_pred = np.arange(0,len(predictions),1)

plt.scatter(test_data.index,predictions,marker="x",c="blue",label="Predictions")
plt.scatter(test_data.index,test_data_y,marker="d",alpha=0.5,c="red", label="True Test Values")
plt.grid()
plt.title('Prediction Graph \n For Extra Tree According to Number of Max Depth\n (According the Information Gained 1-Day Before)')
plt.ylabel("Global Active Power (KWH)")
plt.xlabel("Dates")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
beingsaved.autofmt_xdate()
plt.figtext(0.99, 0.01, 'Model w.r.t Smallest RMSE Value', horizontalalignment='right')
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//Predictions_Est_ET_Number_of_Depth.png', format='png', dpi=1200, bbox_inches='tight')


#Best prediction for RF
lowest = min(number_of_est_dict_rf.items(), key=lambda x: x[1])
print(lowest)
rf_regression = RandomForestRegressor(n_estimators=lowest[0], random_state=0).fit(train_data_x, train_data_y)
predictions=rf_regression.predict(test_data_x)
beingsaved = plt.figure()
x_ticks_for_pred = np.arange(0,len(predictions),1)

plt.scatter(test_data.index,predictions,marker="x",c="blue",label="Predictions")
plt.scatter(test_data.index,test_data_y,marker="d",alpha=0.5,c="red", label="True Test Values")
plt.grid()
plt.title('Prediction Graph \n For Random Forest According to Number of Max Depth\n (According the Information Gained 1-Day Before)')
plt.ylabel("Global Active Power (KWH)")
plt.xlabel("Dates")
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
beingsaved.autofmt_xdate()
plt.figtext(0.99, 0.01, 'Model w.r.t Smallest RMSE Value', horizontalalignment='right')
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//Predictions_Est_RF_Number_of_Depth.png', format='png', dpi=1200, bbox_inches='tight')


beingsaved = plt.figure()
plt.plot(x_mses, et_mses,label='ExtraTreeRegression')
plt.plot(x_mses, rf_mses,label='RandomForestRegression')
plt.ylabel('RMSE')
plt.xlabel('Number of Max Depth')
plt.title('RMSE According to Max Depth Number')
plt.grid()
plt.legend()
plt.show()
beingsaved.savefig('D://Projeler//Energy_Prediction//Codes//RMSE_Max_Depth.png', format='png', dpi=1200,bbox_inches='tight')
#------------------------MAKE PREDICTION FOR ONE DAY (GLOBAL VOLTAGE POWER)-----------------------

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      




