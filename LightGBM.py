import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score,confusion_matrix
from lightgbm import LGBMClassifier
import hyperopt
from hyperopt import hp
import joblib
import logging


# 设置日志级别
logging.getLogger('hyperopt').setLevel(logging.WARNING)  # 设置为INFO级别，也可以选择DEBUG或WARNING



test_size = 0.4    #此处代表训练集和验证集是以(1-test_size:test_size)划分的
seed = 13254   

data01 = pd.read_csv(r'C:\Users\Sengoku\GuanzhongPlain\PointData.csv', encoding='gbk')
filename = r'C:\Users\Sengoku\GuanzhongPlain\06-09\seed{}_{}.txt'.format(seed,test_size)
model_path = r'C:\Users\Sengoku\GuanzhongPlain\model\seed{}_{}.model'.format(seed,test_size)
max_evals = 1000    



header = data01.columns.tolist()
# 使用loc函数,将数据按照target分为两个数据框
data_0 = data01.loc[data01['target'] == 0]
data_1 = data01.loc[data01['target'] == 1]

# 定义随机数的种子,并且设置需要的比列

data_0_X = data_0.drop(columns=["target"], axis=1)
data_0_Y = data_0.target

train_0_X, valid_0_X, train_0_y, valid_0_y = train_test_split(data_0_X, data_0_Y, test_size = test_size,random_state=seed)

save_TrainDate_0 = pd.DataFrame(np.column_stack([train_0_X, train_0_y]), columns=header)
save_ValidDate_0 = pd.DataFrame(np.column_stack([valid_0_X, valid_0_y]), columns=header)

#对target = 1 的数据,划分为1-test_size:test_size,其中1-test_size为train set
data_1_X = data_1.drop(columns=["target"], axis=1)
data_1_Y = data_1.target


train_1_X, valid_1_X, train_1_y, valid_1_y = train_test_split(data_1_X, data_1_Y, test_size = test_size,random_state=seed)

save_TrainDate_1 = pd.DataFrame(np.column_stack([train_1_X, train_1_y]), columns=header)
save_ValidDate_1 = pd.DataFrame(np.column_stack([valid_1_X, valid_1_y]), columns=header)


#合并两个样本的训练集,并打乱数据顺序(避免造成结果异常)
train_date = pd.concat([save_TrainDate_0, save_TrainDate_1])
train_date = train_date.sample(frac=1, random_state=42)
#合并两个样本的测试集,并打乱数据顺序(避免造成结果异常)
valid_date = pd.concat([save_ValidDate_0, save_ValidDate_1])
valid_date = valid_date.sample(frac=1, random_state=42)
#从训练集里选取train_X及train_y
train_y = train_date.target
train_X = train_date.drop(columns=["target"], axis=1)

#从测试集里选取valid_X及valid_y
valid_y = valid_date.target
valid_X = valid_date.drop(columns=["target"], axis=1)




def cross_validation(model_params):
    gbm = LGBMClassifier(**model_params)
    gbm.fit( train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], verbose = 0)
    best_score = gbm.best_score_['valid_1']['auc']
    return 1-best_score

def hyperopt_objective(params):
    cur_param = {
        'objective' : 'binary',
        'boosting_type':params['boosting_type'],
        'metric' : 'auc',
        'num_leaves' : params['num_leaves'],
        'learning_rate' : params['learning_rate'],
        'early_stopping_rounds' : 10,
        'bagging_freq' : params['bagging_freq'],
        'bagging_fraction': params['bagging_fraction'],
        'feature_fraction' : params['feature_fraction']
    }
    print("*" * 30)
    res = cross_validation(cur_param)
    print(params)
    print("Current best 1-auc score is:  {}, auc score is:{} ".format(res,1-res))
    return res   # as hyperopt minimises

params_space = {
    'objective': 'binary',
    "boosting_type":hp.choice("boosting_type",['gbdt','dart','rf']),
    'metric': 'auc',
    "num_leaves":hp.choice("num_leaves",range(15,128)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'bagging_freq': hp.choice("bagging_freq",range(4,7)),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.9),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.9)
}


trials = hyperopt.Trials()

import warnings
warnings.filterwarnings("ignore")

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals = max_evals,
    trials=trials)

print("最佳参数")
print(best)


# 获取最佳参数
best_params = hyperopt.space_eval(params_space, best)
# 添加额外的参数
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['num_iterations'] = 500
best_params['early_stopping_rounds'] = 200

light_model = LGBMClassifier(**best_params)
light_model.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)])

#保存模型
joblib.dump(light_model, model_path)
print("Model saved:", model_path)




y_pred1 = light_model.predict_proba(valid_X)[:, 1]

auc1 = roc_auc_score(valid_y, y_pred1)
y_pred1 = (y_pred1 >= 0.5) * 1
a =  confusion_matrix(valid_y, y_pred1)
a = a.tolist()
a0 = str(a[0])
a1 = str(a[1])

Precesion = str('Precesion: %.4f' % metrics.precision_score(valid_y, y_pred1))
Recall = str('Recall: %.4f' % metrics.recall_score(valid_y, y_pred1))
F1_score = str('F1-score: %.4f' % metrics.f1_score(valid_y, y_pred1))
Accuracy = str('Accuracy: %.4f' % metrics.accuracy_score(valid_y, y_pred1))
AUC = str('AUC: %.4f' % auc1)
AP = str('AP: %.4f' % metrics.average_precision_score(valid_y, y_pred1))
Log_loss = str('Log_loss: %.4f' % metrics.log_loss(valid_y, y_pred1, eps=1e-15, normalize=True, sample_weight=None, labels=None))
kappa_score = str('Kappa_score: %.4f' % metrics.cohen_kappa_score(valid_y, y_pred1))
confusion_matrix = f'{a0}\n{a1}\n'
metrics = f'{AUC}\n{Precesion}\n{Recall}\n{F1_score}\n{Accuracy}\n{AP}\n{Log_loss}\n{kappa_score}\n'

# 以下为特征重要性处理区域
my_dict = dict(zip(train_X.columns, light_model.feature_importances_ ))
# 按值从大到小排序的新字典
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# 计算值的总和
total = sum(sorted_dict.values())
# 计算每个值的百分比，并保存到新字典
dict1 = {key: (value / total) * 100 for key, value in sorted_dict.items()}



with open(filename, 'w') as f:
    # 写入混淆矩阵
    f.write('---------混淆矩阵---------\n')
    f.write(confusion_matrix)

    # 写入评估指标
    f.write('--------评价标准-----------\n')
    f.write(metrics)
    # 写入重要性
    f.write('-------importance------------\n')

    for key, value in dict1.items():
        f.write(f'{key}: {value:.2f}\n')
    # 写入最佳参数,方便后续调用
    f.write('----------best_parm-----------\n')
    f.write(str(best_params))
    f.write('\n')
    seed_str = f'seed = {seed}'
    f.write('----------seed-----------\n')
    f.write(seed_str)
