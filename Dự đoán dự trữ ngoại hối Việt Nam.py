import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lazypredict.Supervised import  LazyRegressor
from sklearn.ensemble import ExtraTreesRegressor

#Lấy data
data=pd.read_excel(r"C:\Users\DELL\Downloads\dự đoán.xlsx")

#Chia bộ dữ liệu
x=data.drop(["time","Dự trữ ngoại hối"],axis=1)
y=data["Dự trữ ngoại hối"]
train_ratio=0.8
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=train_ratio,random_state=40)

#Sử dụng mô hình để dự đoán
reg = ExtraTreesRegressor()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

#Trực quan hóa mô hình dự đoán so sánh với dữ liệu thực tế
time = data['time']
data['y_pred'] = reg.predict(x)
plt.figure(figsize=(12,6))
plt.plot(time, y, label='Thực tế', linewidth=2)
plt.plot(time, data['y_pred'], label='Dự đoán', linestyle='--')
plt.xlabel("Thời gian")
plt.ylabel("Dự trữ ngoại khối")
plt.title("Dự trữ ngoại khối: Thực tế vs Dự đoán theo thời gian")
plt.legend()
plt.grid(True)
plt.show()