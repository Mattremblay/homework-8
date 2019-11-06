from sklearn.datasets import load_wine
from numpy import shape
from numpy import polyfit, polyval
import pandas as pd
import matplotlib.pyplot as plt
wine_data = load_wine()
dir (wine_data)
df = pd.DataFrame(data=wine_data.data, columns = (wine_data.feature_names))
# print(df.head())
# print(wine_data.DESCR)
# type(wine_data.data))
# print(shape(wine_data.data))
# print(wine_data.feature_names)
# print(wine_data.target)
# df['total_phenols']
# print(df.describe())
# plt.hist(df['total_phenols'])
# plt.xlabel('total_phenols')
# plt.ylabel('count')
flava = df['flavanoids']
color = df['color_intensity']
p = polyfit(flava, color, 1)
plt.plot(flava, color,'o')
plt.plot(sorted(flava), polyval(p, sorted(flava)),'-')
# plt.scatter(df['total_phenols'],(df['color_intensity']))
plt.ylabel('flavanoids', size = 15)
plt.xlabel('color_intensity', size = 15)
plt.show()

print(df.corr()['flavanoids'])