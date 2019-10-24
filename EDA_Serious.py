#!/usr/bin/env python
# coding: utf-8

# ### 測試
# #### 根據資料，分析分布與創建新的欄位用來描述: 
# 1. 在所有城市之中，每個城市與其他的城市關係
# 2. 對這個帳戶來說，這個城市與其他城市的關係
# 3. 這個日期與其他日期的關係(考慮周或年)
# 4. 這筆消費與所有其他消費的關係
# 5. 對這個帳戶來說，這筆消費與其他消費的關係
# 
# #### Note
# - 以帳戶來分析應優先於全域
# - 分布要考慮資料平衡
# - 不要對類別資料算相關性
# 
# 

# In[1]:


import pandas as pd
ori_df = pd.read_csv('train.csv')

### 看有幾個column
print(f'-共有{ori_df.shape[0]}筆資料')
#print('-共有{0}筆資料'.format(df.shape[0]))
print('-分布於%d欄'%ori_df.shape[1])
import numpy as np
def timeSinCosTrans(df):

    Time = df.loc[:,'loctm'].values
    # // 是整除的意思
    s, m, h = [(Time//100**(i)%100) for i in range(3)]
    Time = h*3600+m*60+s

    dfTime = pd.DataFrame()
    dfTime.loc[:,'TIME_SIN'] = np.sin(2*np.pi*Time/(24*60*60))
    dfTime.loc[:,'TIME_COS'] = np.cos(2*np.pi*Time/(24*60*60))

    df.drop(columns = 'loctm', inplace =True)
    df = pd.concat([df, dfTime], axis = 1)

    return df


# In[2]:


def test_validWithAll(testModdf, y_valid, X_valid):
     
    y_test = testModdf['fraud_ind']
    y_test = pd.concat([y_test, y_valid], axis = 0)

    data = testModdf.drop(columns = 'fraud_ind')
    # 合併X_valid和其他沒被詐騙的資料
    data = pd.concat([data,X_valid], axis = 0, )
    #data.info()


    y_true = y_test.values
    y_pred = model.predict(data)

    display_score(y_true, y_pred)


# In[3]:


def display_score(y_true, y_pred):
    print('F1 Score:   %s'%(metrics.f1_score(y_true, y_pred, average=None)))
    print('...\nconfusion_matrix:   \n%s'%(metrics.confusion_matrix(y_true, y_pred))) 
    print('...\nprecision_recall_fscore_support: ')
    print(metrics.precision_recall_fscore_support(y_true, y_pred))
    print('...\nauc:   ')
    print(metrics.roc_auc_score(y_true, y_pred))


# In[4]:


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
import matplotlib.pyplot as plt


# In[11]:


#抓出所有被詐騙過的帳戶資料進行分析
# 

df = ori_df.copy()
YN = ['ecfg','flbmk', 'flg_3dsmk', 'insfg', 'ovrlt']
for col in YN:
    df.loc[df[col] =='N', col] = 0
    df.loc[df[col] =='Y', col] = 1
df.fillna(0, inplace =True)

df = df.sample(frac=1)

fraud_df = df.loc[df['fraud_ind'] == 1]
print(len(fraud_df))
fraud_bacno_list = list(set(fraud_df['bacno']))

new_df = df[df['bacno'].isin(fraud_bacno_list)]
print(len(new_df))


# In[12]:


import seaborn as sns

len(new_df)# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(new_df['fraud_ind'].value_counts()[0]/len(new_df) * 100,2), '% of the dataset')
print('Frauds', round(new_df['fraud_ind'].value_counts()[1]/len(new_df) * 100,2), '% of the dataset')

colors = ["#0101DF", "#DF0101"]

print('Distribution of the Classes in the subsample dataset')
print(new_df['fraud_ind'].value_counts()/len(new_df))

sns.countplot('fraud_ind', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# ### 相關性如何分析

# In[13]:


import pandas as pd
data_array = pd.DataFrame({"支持湖人": [36, 40, 50], "支持勇士": [20, 19, 15]},["社區A", "社區B", '社區C'])

data_array

# method : {'pearson', 'kendall', 'spearman'}
#data_array.corr()


# In[14]:


data_array = pd.DataFrame({"湖人支持度排名": [1, 2, 1], "勇士支持度排名": [2, 3, 30]},["社區A", "社區B", '社區C'])
data_array
# data_array.corr()
data_array.corr(method = 'pearson') #This is as same as the upper
data_array.corr(method='kendall')
#data_array.corr(method='spearman')


# ### 相關性分析參考 
# http://www.r-web.com.tw/guider/1/section_B_example_Q1.php
# 
# 在探討兩變數間的相關性時，依據變數型態的不同，分析的方法也會不同。通常變數區分為連續型變數與類別型變數，變數的型態主要是由資料的 **測量尺度(scale of measurement)** 來決定，統計學中概分為 __四種測量尺度__ ，名目尺度(nominal scale)、順序尺度(ordinal scale)、區間尺度(interval scale)與比例尺度(ratio scale)。
# 
# - 名目尺度：數字只是一個代號，並不具測量的意義，如以0代表女性，1代表男性，0與1之間無法運算。
# - 順序尺度：可依相對大小來排序，並給予對應的數字，此數字僅代表測量值的差異，但無法表示出差異的強度；如顧客滿意度區分為1至4，1為最差4為最佳。
# - 區間尺度：測量值的數字可表示出差異的距離，但此數字無法計算比值，如時間的差異，9點與10點有1個小時的差距，但無法說出倍數的關係。
# - 比例尺度：此種尺度的數字可以做任何的計算，0是一個絕對的原點，各種體積與面積的計算皆屬於此種尺度。
# 
# 上述四種尺度中:
# - 名目與順序尺度的資料型態為類別型變數
# - 而區間與比例尺度的資料型態則為連續型變數
# - 當兩變數皆為連續型變數時，適合以皮爾森(Pearson)相關分析來計算其相關係數之高低
# - 當兩變數皆為類別型變數時，資料為順序關係可以斯皮爾曼等級相關(Spearman rank correlation)來了解變數間的相關性
# - 當資料為順序尺度時則適合:
#     1. 肯德爾等級相關性(Kendall tau)檢定
# - 當資料為名目尺度時則適合:
#     1. 費雪(R. A. Fisher)的exact檢定分析
#     2. 卡方的獨立性(independence)與齊一性(homogeneity)檢定
#     

# ### pandas.DataFrame.corr
# 1. [panda API](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
# 2. [維基百科- 檢定](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%85%B3)
# 
# - method : {‘pearson’, ‘kendall’, ‘spearman’}
#     1. pearson 皮爾森相關係數 **(當兩變數皆為連續型變數時)**
#     2. kendall 肯德爾等級相關性 **(衡量兩個人為次序尺度變數（原始資料為等距尺度）之相關性。**
#     3. spearman 斯皮爾曼相關係數 **(衡量兩個次序尺度變數之相關性)**

# In[15]:


new_df.corr(method='pearson')


# ### 所有資料的相關性分析
# 相關性分析是為了對假設作檢定

# 先對全體資料做相關性分析，再對少量資料做相關性分析，將兩個分析結果基於無母數進行檢定。可以針對資料的刪減是否會影響相關性的虛無假設作檢定。

# In[39]:


#new_df 是一個平衡資料集
#包含where info的feature包括: stocn scity hcefg
where_ls = ['stocn', 'scity', 'hcefg']
where_ls = ['stocn', 'scity']
df = pd.DataFrame()
#取得只包含位置資訊的資料
for i in where_ls:
    df = pd.concat([df, new_df[i]], axis = 1)

lis = []
lisa = list(set(df['stocn']))

# 是否輸出影像
tmp_key = 1     
    
for i in lisa:
    lisb = list(set(df[df['stocn'] == i]['scity']))
    if tmp_key:        
        print('國家\'%d\'有%d個城市，如下所示:'%(i, len(lisb)))
        print(lisb)
    lis.append([i,len(lisb)])
    #print(len(lisb))
# lis [[stocn_id, appear numbers]... ]
new_df.columns
# a = new_df[new_df['scity'] == 0]['fraud_ind'].sum()
# b = len(new_df[new_df['scity'] == 0]['fraud_ind'])
print(len(lis))


# In[ ]:





# In[ ]:




