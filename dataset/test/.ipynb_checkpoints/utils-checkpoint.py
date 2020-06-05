import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def score(y_true, y_pred):
# y_true = y_train
# y_pred = clf.predict(X_train)
    print('---')
    print('R2, ', r2_score(y_true, y_pred))
    print('mse, ', mse(y_true, y_pred))
    print('mae, ', mae(y_true, y_pred))
    print('rmse, ', mse(y_true, y_pred) ** 0.5)
    print('---')
    
def XGB_def():
    xgbreg = xgb.XGBRegressor(
    #樹的個數
    n_estimators=1000,
    # 如同學習率
    learning_rate= 0.3, 
    # 構建樹的深度，越大越容易過擬合    
    max_depth=6, 
    # 隨機取樣訓練樣本 訓練例項的子取樣比
    subsample=1, 
    # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子
    gamma=0, 
    # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。
    reg_lambda=1,  
    #隨機種子
    seed=1000, 
    # L1 正則項引數
#        reg_alpha=0,
    #設定成1則沒有執行資訊輸出，最好是設定為0.是否在執行升級時列印訊息。
    silent=0 ,
    tree_method = 'gpu_hist', 
    predictor = 'gpu_predictor',
    eval_metric= 'mse',
    objective = 'reg:squarederror'
    )
    
    return xgbreg

# 定義xgb
def XGB_train_score(X_train, y_train):
    # 定義模型
    xgbreg = XGB_def()
    
    # 模型 訓練
    xgbreg.fit(X_train,y_train,eval_metric='rmse')

    # 預測值
    y_pred=xgbreg.predict(X_train)

    # 評估
    score(y_train, y_pred)
    return xgbreg

def Moving_average(data, window_size, stride):
    X = []
    y = []
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size].mean())
#         y.append(data[i:i+window_size].mean())
        i += stride
    X = np.array(X)
#     y = np.array(y)
#     assert len(X) == len(y)
    return X
# ym = Moving_average(Xm, 7, 1)
# ym.shape