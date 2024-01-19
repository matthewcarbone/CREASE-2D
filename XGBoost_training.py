import pandas as pd
import numpy as np
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.utils import shuffle
#comment the cupy import line if you run on CPU's
import cupy as cp
import warnings
warnings.simplefilter('ignore')

#access the dataset 
df = pd.read_csv('/content/drive/MyDrive/CREASE-2D/Crease_2400_126/train_dataset_2400_126.csv')
df_shuffled = shuffle(df, random_state=189)
X = df_shuffled.drop(columns=['I_q', 'Sample ID'])
y = df_shuffled['I_q']
#y_norm, min_val, max_val = normalize_target(y)
#load the data into GPU arrays, comment the below lines if you run on CPUs 
X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)
#define the parameter space to get the optimized values using Bayesian optimization
param_space = {
    'n_estimators': np.arange(50, 1000, 50),
    'max_depth': np.arange(3, 15),
    'learning_rate': np.arange(0.001, 0.1, 0.001),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'gamma': np.arange(0, 1, 0.1),
    'min_child_weight': np.arange(1, 10),
    'reg_lambda': np.arange(0.1, 1, 0.1),
    'reg_alpha': np.arange(0.1, 1, 0.1),
    'colsample_bylevel': np.arange(0.5, 1.0, 0.1)
}

#initialize the XGBoost model, remove device = 'cuda' if you run on CPU's
xgb_reg = xgb.XGBRegressor(tree_method='hist', importance_type='cover',device='cuda', random_state=51)

#We use Skopt library to tume the parameter space
opt = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    return_train_score=True,
    refit=False,
    optimizer_kwargs={'base_estimator': 'GP'}
)
#Train the model, change X_gpy to X and y_gpu to y if running on CPUs
opt.fit(X_gpu.get(), y_gpu.get())
best_params = opt.best_params_
best_score = opt.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

#Train the XGBoost model with tuned hyperparameters on CPUs  
final_xgb = xgb.XGBRegressor(**best_params, tree_method='hist', importance_type='cover', random_state=51)
final_xgb.fit(X,y)
#get the weights assigned to each feature as cover method type
cover_importance = final_xgb.feature_importances_
print("Feature importance weights:", cover_importance)
#edit the path to save it to desired location
final_xgb.save_model('desired_location/xgbmodel_2400_126.json')