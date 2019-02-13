# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:07:49 2018

@author: t4nis
"""

import pandas as pd
import json

"""
data1 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_1.json')
df1=pd.DataFrame(data=data1)

data2 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_2.json')
df2=pd.DataFrame(data=data2)
"""

data3 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_3.json')
df3=pd.DataFrame(data=data3)

data4 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_4.json')
df4=pd.DataFrame(data=data4)

data5 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_5.json')
df5=pd.DataFrame(data=data5)

data6 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_6.json')
df6=pd.DataFrame(data=data6)

data7 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_7.json')
df7=pd.DataFrame(data=data7)

data8 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_8.json')
df8=pd.DataFrame(data=data8)

data9 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_9.json')
df9=pd.DataFrame(data=data9)

data10 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_10.json')
df10=pd.DataFrame(data=data10)

data11 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_2_11.json')
df11=pd.DataFrame(data=data11)

data12 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_1.json')
df12=pd.DataFrame(data=data12)

data13 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_2.json')
df13=pd.DataFrame(data=data13)

data14 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_3.json')
df14=pd.DataFrame(data=data14)

data15 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_4.json')
df15=pd.DataFrame(data=data15)

data16 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_5.json')
df16=pd.DataFrame(data=data16)

data17 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_6.json')
df17=pd.DataFrame(data=data17)

data18 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_7.json')
df18=pd.DataFrame(data=data18)

data19 = pd.read_json('/Users/t4nis/Desktop/BIA 660/Project/Extracted_data_3_8.json')
df19=pd.DataFrame(data=data19)




frames = [df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
data=pd.concat(frames)
#print (data)

data.to_csv("Data.csv",sep=',')