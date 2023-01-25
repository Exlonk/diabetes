
#%% 
# Librerias
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import callback,Dash, html, dcc
from jupyter_dash import JupyterDash
from dash_bootstrap_templates import load_figure_template # para los fondos de las imagenes 
import os                       
from importlib.machinery import SourceFileLoader
ds = SourceFileLoader("add",os.path.join(os.path.dirname(__file__),'data_science.py')).load_module()           
#%% [Init]
# Functions

path_models= os.path.join(os.path.dirname(__file__),'models')
path_validation_curves = os.path.join(os.path.dirname(__file__),'validation_curves')
path_figures = os.path.join(os.path.dirname(__file__),'figures')
path_dataframes = os.path.join(os.path.dirname(__file__),'dataframes')

# Layout style

load_figure_template("cosmo")

discrete_color_graph = px.colors.sequential.Plasma

# End

# ----------------------------- DATA INSIGHTS -------------------------------- #

# Pure Raw Dataframe 

X_raw, y_raw = datasets.load_diabetes(return_X_y=True,as_frame=True)

X_raw.columns = ['age', 'sex', 'body mass index','average blood pressure',
                 'total serum cholesterol','low-density lipoproteins', 
                 'hight-density lipoproteins','total cholesterol',
                 'possibly log of serum triglycerides level',
                 'blood sugar lever']

y_raw.columns = ['target (progression one year after baseline)']

all_features = ['age', 'sex', 'body mass index','average blood pressure',
                       'total serum cholesterol','low-density lipoproteins', 
                       'hight-density lipoproteins','total cholesterol',
                        'possibly log of serum triglycerides level',
                        'blood sugar lever']

df_raw = pd.concat([X_raw,y_raw],axis=1)

table = ds.table(df_raw,textcolor='#373a3c') 

shape = ds.load_plot_json(path_figures,'shape') 

info_fig = ds.load_plot_json(path_figures,'info_fig') 

# Duplicates and Missing values

duplicates = ds.load_plot_json(path_figures,'duplicates') 

missing_values = ds.load_plot_json(path_figures,'missing_values')

# Histograms

histograms = []
for i in range(0,11):
    histograms.append(ds.load_plot_json(path_figures,'histogram_'+str(i)))

# Outliers 

box_plot_pure = ds.load_plot_json(path_figures,'box_plot_pure')
box_plot_lof = ds.load_plot_json(path_figures,'box_plot_ifo')
box_plot_ifo = ds.load_plot_json(path_figures,'box_plot_lof')

# ---------------------------- FEATURE SELECTION ----------------------------- #

# Correlation

pearson_correlation = ds.load_plot_json(path_figures,'pearson_correlation')
pearson_correlation.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
spearman_correlation = ds.load_plot_json(path_figures,'spearman_correlation')
spearman_correlation.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

mutual_info_correlation = ds.load_plot_json(path_figures,'mutual_info_correlation')
mutual_info_correlation.update_yaxes(range=(0,0.30))

mutual_info_correlation.update_layout(title='Numerical and Categorical Correlation using Mutual Info Regression')
aic_correlation = ds.load_plot_json(path_figures,'aic_correlation')
aic_correlation.update_yaxes(range=(-250,450))

# <------------------------- Train/CV/Test sets -----------------------------> #

X_train = pd.read_csv(os.path.join(path_dataframes,'X_train.csv'))
X_cv = pd.read_csv(os.path.join(path_dataframes,'X_cv.csv'))
X_test = pd.read_csv(os.path.join(path_dataframes,'X_test.csv'))
y_train = pd.read_csv(os.path.join(path_dataframes,'y_train.csv'))
y_cv = pd.read_csv(os.path.join(path_dataframes,'y_cv.csv'))
y_test = pd.read_csv(os.path.join(path_dataframes,'y_test.csv'))

numerical_data = list(X_train.columns[:-2])
categorical_data = list(X_train.columns[-2:])

# <-------------------------- MACHINE LEARNING MODELS -----------------------> #

# <--------------------------- Sample Distribution --------------------------> #

sample_distribution_fig = ds.load_plot_json(path_figures,'sample_distribution_fig')

# <--------------------------- Baseline Model -------------------------------> #

baseline_accuracy_train,baseline_accuracy_cv,baseline_accuracy_test,ind_baseline_train,\
ind_baseline_cv,ind_baseline_test, baseline_bias_variance = \
      ds.load_plot_json(path_figures,'baseline_accuracy_train'),ds.load_plot_json(path_figures,'baseline_accuracy_cv'), \
      ds.load_plot_json(path_figures,'baseline_accuracy_test'),ds.load_plot_json(path_figures,'baseline_ind_train'), \
      ds.load_plot_json(path_figures,'baseline_ind_cv'),ds.load_plot_json(path_figures,'baseline_ind_test'), \
      ds.load_plot_json(path_figures,'baseline_bias_variance')


baseline_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_baseline_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
baseline_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_baseline_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
baseline_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_baseline_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
baseline_bias_variance.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

# <--------------------------- Polynomial Model -----------------------------> #

# print('Entering... Polynomial')

# Degree optimization

mse_train_cv_rpoly_graph = ds.load_plot_json(path_figures,'mse_train_cv_rpoly_graph')
r2_train_cv_rpoly_graph = ds.load_plot_json(path_figures,'r2_train_cv_rpoly_graph')
mse_train_cv_rpoly_graph.update_layout(title='Ridge Polynomial Regression Degrees')
mse_train_cv_rpoly_graph.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
r2_train_cv_rpoly_graph.update_layout(title='Ridge R2-Squared Training/Cv Set')
r2_train_cv_rpoly_graph.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

mse_train_cv_poly_graph = ds.load_plot_json(path_figures,'mse_train_cv_poly_graph')
r2_train_cv_poly_graph = ds.load_plot_json(path_figures,'r2_train_cv_poly_graph')
mse_train_cv_poly_graph.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
r2_train_cv_poly_graph.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

linear_regression_accuracy_train = ds.load_plot_json(path_figures,'linear_regression_accuracy_train')
linear_regression_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_linear_regression_accuracy_train = ds.load_plot_json(path_figures,'ind_linear_regression_accuracy_train')
ind_linear_regression_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

linear_regression_accuracy_cv = ds.load_plot_json(path_figures,'linear_regression_accuracy_cv')
linear_regression_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_linear_regression_accuracy_cv = ds.load_plot_json(path_figures,'ind_linear_regression_accuracy_cv')
ind_linear_regression_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

linear_regression_accuracy_test = ds.load_plot_json(path_figures,'linear_regression_accuracy_test')
linear_regression_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_linear_regression_accuracy_test = ds.load_plot_json(path_figures,'ind_linear_regression_accuracy_test')
ind_linear_regression_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

linear_regression_bias_variance = ds.load_plot_json(path_figures,'linear_regression_bias_variance')
linear_regression_bias_variance.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

# Main Model
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_regression
# import joblib
# poly_model = Ridge(alpha=27)
# poly_features = PolynomialFeatures(degree=10, include_bias=False)
# X_train_poly = poly_features.fit_transform(X_train[numerical_data])
# X_train_poly = pd.DataFrame(X_train_poly,columns=poly_features.get_feature_names_out())
# X_train_poly =pd.concat([X_train_poly,X_train[categorical_data]],axis=1)
# X_cv_poly = poly_features.fit_transform(X_cv[numerical_data])
# X_cv_poly = pd.DataFrame(X_cv_poly,columns=poly_features.get_feature_names_out())
# X_cv_poly =pd.concat([X_cv_poly,X_cv[categorical_data]],axis=1)

# X_test_poly = poly_features.fit_transform(X_test[numerical_data])
# X_test_poly = pd.DataFrame(X_test_poly,columns=poly_features.get_feature_names_out())
# X_test_poly =pd.concat([X_test_poly,X_test[categorical_data]],axis=1)
# poly_model.fit(X_train_poly,y_train)
# y_pred_train_poly =  poly_model.predict(X_train_poly)
# y_pred_cv_poly =  poly_model.predict(X_cv_poly)
# y_pred_test_poly =  poly_model.predict(X_test_poly)


poly_accuracy_train,poly_accuracy_cv,poly_accuracy_test,ind_poly_train,\
ind_poly_cv,ind_poly_test, poly_bias_variance \
 = ds.load_plot_json(path_figures,'poly_accuracy_train'),ds.load_plot_json(path_figures,'poly_accuracy_cv'), \
   ds.load_plot_json(path_figures,'poly_accuracy_test'),ds.load_plot_json(path_figures,'poly_ind_train'), \
   ds.load_plot_json(path_figures,'poly_ind_cv'),ds.load_plot_json(path_figures,'poly_ind_test'), \
   ds.load_plot_json(path_figures,'poly_bias_variance')

poly_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_poly_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
poly_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_poly_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
poly_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
ind_poly_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
poly_bias_variance.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

learning_curve = ds.load_plot_json(path_figures,'learning_curve')
learning_curve.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
learning_curve.update_yaxes(range=(0,1))
# KNeighbor Regresor ---------------------------------------------------------->
from sklearn.neighbors import KNeighborsRegressor
# import joblib
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV

# print('Entering... KNN')

# Parameters KNN

knn_mode = 'design'
knn_save_model = True

# Validation Curves KNN

validation_parameters_knn = {
    'n_neighbors' : [i for i in range(5,100,5)],
    'weights': ['uniform','distance'],
    'algorithm': ['auto','ball_tree','kd_tree','brute'],
    'leaf_size': [i for i in range(30,600,30)], 
    'p': [i for i in range(1,11)],
}
list_validation_curve_knn = ds.validation_curves(KNeighborsRegressor(),X_train,y_train,X_cv,y_cv,cv=5,
   scoring='neg_mean_absolute_error',path_ = path_validation_curves,original_or_positive='positive',
   parameters_range=validation_parameters_knn,save=True,color_line=['#fdca26','#0d0887']) # 

list_validation_curve_knn_figures = ["knn_val_curve_1","knn_val_curve_2",
                                    "knn_val_curve_3","knn_val_curve_4",
                                    "knn_val_curve_5"]
conter_figures_knn = 0
for i in list_validation_curve_knn_figures:
    vars()[i]= list_validation_curve_knn[conter_figures_knn]
    conter_figures_knn+=1

# Model KNN
# if knn_mode == 'train':

#     parameters_knn_model = {
#         'n_neighbors' : [23],
#         'weights': ['uniform'],
#         'p': [1],
#         'leaf_size': [30],
#         'algorithm':['auto']
#     }

#     knn_regressor = KNeighborsRegressor()
#     knn_model=GridSearchCV(knn_regressor,param_grid=parameters_knn_model,\
#                             scoring='r2',n_jobs=-1)

#     knn_model.fit(X_train,y_train)
#     y_pred_train_knn =  knn_model.predict(X_train)
#     y_pred_cv_knn =  knn_model.predict(X_cv)
#     y_pred_test_knn =  knn_model.predict(X_test)

#     if knn_save_model == True:
#         joblib.dump(knn_model,os.path.join(path_models,'knn.joblib'))

# elif knn_mode == 'design':
#     knn_model = joblib.load(os.path.join(path_models,'knn.joblib'))
#     y_pred_train_knn =  knn_model.predict(X_train)
#     y_pred_cv_knn =  knn_model.predict(X_cv)
#     y_pred_test_knn =  knn_model.predict(X_test)


knn_accuracy_train,knn_accuracy_cv,knn_accuracy_test,knn_ind_train,\
knn_ind_cv,knn_ind_test, knn_bias_variance \
 = ds.load_plot_json(path_figures,'knn_accuracy_train'),ds.load_plot_json(path_figures,'knn_accuracy_cv'), \
   ds.load_plot_json(path_figures,'knn_accuracy_test'),ds.load_plot_json(path_figures,'knn_ind_train'), \
   ds.load_plot_json(path_figures,'knn_ind_cv'),ds.load_plot_json(path_figures,'knn_ind_test'), \
   ds.load_plot_json(path_figures,'knn_bias_variance')

knn_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_ind_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_ind_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_ind_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
knn_bias_variance.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
# Support Vector Machine ------------------------------------------------------>

from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV
# import joblib

# print('Entering... SVM')

# Parameters SVM

svm_mode = 'design'
svm_save_model = True

# validation curves SVM

validation_parameters_svm = {
            'kernel':['sigmoid','linear','poly','rbf'],'gamma':['scale','auto'],
            'degree':[1,2,3,4,5,6,7,8,9,10],'coef0':[0,1,2,3,4,5,10,20,30,40,50,100]}

list_validation_curve_svm = ds.validation_curves(SVR(),X_train,y_train,X_cv,y_cv,cv=5,
   scoring='neg_mean_absolute_error',path_ = path_validation_curves,original_or_positive='positive',
   parameters_range=validation_parameters_svm,save=False,color_line=['#fdca26','#0d0887']) # 

list_validation_curve_svm_figures = ["svm_val_curve_1","svm_val_curve_2",
                        "svm_val_curve_3","svm_val_curve_4" ]
conter_figures_svm = 0

for i in list_validation_curve_svm_figures:
    vars()[i]= list_validation_curve_svm[conter_figures_svm]
    conter_figures_svm+=1

# Main Model SVM

# if svm_mode == 'train':

#     svm_regressor = SVR()

#     parameters_svm_model = {
#             'kernel':['linear','poly'],'gamma':['scale','auto'],
#             'degree':[1,2,3]}

#     svm_model=GridSearchCV(svm_regressor,param_grid=parameters_svm_model,\
#                             scoring='neg_mean_squared_error',n_jobs=-1)

#     svm_model = svm_model.fit(X_train,y_train.values.ravel())
#     y_pred_train_svm = svm_model.predict(X_train)
#     y_pred_cv_svm = svm_model.predict(X_cv)
#     y_pred_test_svm = svm_model.predict(X_test)

#     if svm_save_model == True:
#         joblib.dump(svm_model,os.path.join(path_models,'svm.joblib'))

# elif svm_mode == 'design':
#     svm_model = joblib.load(os.path.join(path_models,'svm.joblib'))

#     y_pred_train_svm =  svm_model.predict(X_train)
#     y_pred_cv_svm =  svm_model.predict(X_cv)
#     y_pred_test_svm =  svm_model.predict(X_test)

svm_accuracy_train,svm_accuracy_cv,svm_accuracy_test,svm_ind_train,\
svm_ind_cv,svm_ind_test, svm_bias_variance = \
      ds.load_plot_json(path_figures,'svm_accuracy_train'),ds.load_plot_json(path_figures,'svm_accuracy_cv'), \
      ds.load_plot_json(path_figures,'svm_accuracy_test'),ds.load_plot_json(path_figures,'svm_ind_train'), \
      ds.load_plot_json(path_figures,'svm_ind_cv'),ds.load_plot_json(path_figures,'svm_ind_test'), \
      ds.load_plot_json(path_figures,'svm_bias_variance')

svm_accuracy_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_ind_train.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_accuracy_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_ind_cv.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_accuracy_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_ind_test.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')
svm_bias_variance.update_layout(paper_bgcolor="#ffffff",plot_bgcolor='#fff')

# Ensemble Tree --------------------------------------------------------------->

from xgboost import XGBRegressor
# from sklearn.model_selection import RandomizedSearchCV
# import joblib

# print('Entering... Tree')

# Parameters Tree

# tree_mode = 'design' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
# tree_save_model = True # Save the model

# Validation Curves

validation_parameters_tree = {
    "eta": np.linspace(0.01,0.2,20).tolist(),
    "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
    "max_depth"        : [ 1,2,3, 4, 5, 6, 7,8, 9,10, 12, 15],
    "subsample"         : np.linspace(0.01,1,10).tolist(),
    "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
    'seed':[1,2,3,4,5,6,7,8,9,10] 
}

list_validation_curve_tree = ds.validation_curves(XGBRegressor(),X_train,y_train,X_cv,y_cv,cv=5,
   scoring='neg_mean_absolute_error',path_ = path_validation_curves,original_or_positive='positive',
   parameters_range=validation_parameters_tree,save=False,color_line=['#fdca26','#0d0887']) # 

list_validation_curve_tree_figures = ["tree_val_curve_1","tree_val_curve_2","tree_val_curve_3",
                                        "tree_val_curve_4","tree_val_curve_5",'tree_val_curve_6' ]
conter_figures_tree = 0

for i in list_validation_curve_tree_figures:
    vars()[i]= list_validation_curve_tree[conter_figures_tree]
    conter_figures_tree+=1

# # Main Model Tree

# if tree_mode == 'train':

#     tree_regressor = XGBRegressor()

#     parameters_tree_model = {
#          "eta": np.linspace(0.01,0.05,3).tolist(),
#          "max_depth"        : [1,2,3],
#          "subsample"         : np.linspace(0.01,1,3).tolist(),
#      }

#     tree_model=GridSearchCV(tree_regressor,param_grid=parameters_tree_model,\
#                             scoring='neg_mean_squared_error',n_jobs=-1)

#     tree_model.fit(X_train,y_train)
#     y_pred_train_tree = tree_model.predict(X_train)
#     y_pred_cv_tree = tree_model.predict(X_cv)
#     y_pred_test_tree = tree_model.predict(X_test)

#     if tree_save_model == True:
#         joblib.dump(tree_model,os.path.join(path_models,'ensemble_tree.joblib'))

# elif tree_mode == 'design':
#     tree_model = joblib.load(os.path.join(path_models,'ensemble_tree.joblib'))
#     y_pred_train_tree = tree_model.predict(X_train)
#     y_pred_cv_tree = tree_model.predict(X_cv)
#     y_pred_test_tree = tree_model.predict(X_test)

tree_accuracy_train,tree_accuracy_cv,tree_accuracy_test,tree_ind_train,\
tree_ind_cv,tree_ind_test, tree_bias_variance = \
      ds.load_plot_json(path_figures,'tree_accuracy_train'),ds.load_plot_json(path_figures,'tree_accuracy_cv'), \
      ds.load_plot_json(path_figures,'tree_accuracy_test'),ds.load_plot_json(path_figures,'tree_ind_train'), \
      ds.load_plot_json(path_figures,'tree_ind_cv'),ds.load_plot_json(path_figures,'tree_ind_test'), \
      ds.load_plot_json(path_figures,'tree_bias_variance') 


# Neural Network -------------------------------------------------------------->

# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.losses import MeanSquaredError
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import joblib
# from scikeras.wrappers import KerasRegressor
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.constraints import MaxNorm

# print('Entering... Neural Network')

# # Parameters Neural Network

# neural_mode = 'design' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
# neural_save_model = True # Save the model
# neural_network_graph = ds.neural_network_fig([1,5,10,20,40,70,70,70,40,20,10,5,1]) # Graph
neural_network_graph = ds.load_plot_json(path_figures,'neural_network_graph')
# if neural_mode == 'train':
#     l2 = 0.5
#     drop = 0.75
#     neural_model = Sequential([
#     Dense(units=50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dropout(drop),
#     Dense(units=700, activation = 'relu',  kernel_constraint=MaxNorm(3)), 
#     Dropout(drop),
#     Dense(units=700, activation = 'relu',  kernel_constraint=MaxNorm(3)),
#     Dropout(drop),
#     Dense(units=700, activation = 'relu',  kernel_constraint=MaxNorm(3)), 
#     Dense(units=400, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=200, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=100, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=50, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
#     Dense(units=1, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
# ])
    
#     neural_model.build(input_shape=X_train.shape)
#     neural_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = tf.keras.losses.MeanAbsoluteError())

#     neural_model.fit(X_train,y_train.values.ravel(), epochs=300)

#     y_pred_train_network = neural_model.predict(X_train)
#     y_pred_cv_network= neural_model.predict(X_cv)
#     y_pred_test_network= neural_model.predict(X_test)

#     if neural_save_model == True:
#          tf.keras.models.save_model(neural_model,os.path.join(path_models,'neural_network.h5'))

# elif neural_mode == 'design':

#     neural_model = tf.keras.models.load_model(os.path.join(path_models,'neural_network.h5'))

#     y_pred_train_network = neural_model.predict(X_train)
#     y_pred_cv_network= neural_model.predict(X_cv)
#     y_pred_test_network= neural_model.predict(X_test)

network_accuracy_train,network_accuracy_cv,network_accuracy_test,network_ind_train,\
network_ind_cv,network_ind_test, network_bias_variance = \
      ds.load_plot_json(path_figures,'network_accuracy_train'),ds.load_plot_json(path_figures,'network_accuracy_cv'), \
      ds.load_plot_json(path_figures,'network_accuracy_test'),ds.load_plot_json(path_figures,'network_ind_train'), \
      ds.load_plot_json(path_figures,'network_ind_cv'),ds.load_plot_json(path_figures,'network_ind_test'), \
      ds.load_plot_json(path_figures,'network_bias_variance') 

# Model Comparation ----------------------------------------------------------->

# models = ['poly','knn','svm','tree','network']

# # MAE
# from sklearn.metrics import mean_absolute_error
# model_comparation_mse = {}

# vars()['total_mse']=[]
# for k in ['train','cv','test']:
#     for i in models:
#         vars()['total_mse'].append(mean_absolute_error(vars()['y_'+k],vars()['y_pred_'+k+'_'+i]))
        
# model_comparation_mse['values'] = vars()['total_mse']
# model_comparation_mse['set'] = ['train']*5+['cv']*5+['test']*5
# model_comparation_mse['estimator'] = models*3
# model_comparation_mse = pd.DataFrame(model_comparation_mse)

# model_vs_model_mse_graph = px.bar(model_comparation_mse,y='set',x='values',color='estimator',barmode='group',text_auto='.2s',
#                               title='MAE Models Comparation',labels={'estimator':'','values':'Mean Absolute Error','set':''},
#                               color_discrete_sequence=['#0d0887','#9c179e','#ed7953','#fb9f3a','#f0f921'],orientation='h')
# model_vs_model_mse_graph.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

# # R2
# model_comparation_r2= {}

# vars()['total_r2']=[]
# for k in ['train','cv','test']:
#     for i in models:
#         vars()['total_r2'].append(r2_score(vars()['y_'+k],vars()['y_pred_'+k+'_'+i]))
        
# model_comparation_r2['values'] = vars()['total_r2']
# model_comparation_r2['set'] = ['train']*5+['cv']*5+['test']*5
# model_comparation_r2['estimator'] = models*3
# model_comparation_r2 = pd.DataFrame(model_comparation_r2)

# model_vs_model_r2_graph_train=px.pie(model_comparation_r2[model_comparation_r2['set']=='train'],values='values',hole=.5,names='estimator',
#                                      title='Model Comparison in the Training Set <br> <sub>by R2</sub>',
#                                      color_discrete_sequence=['#0d0887','#9c179e','#ed7953','#fb9f3a','#f0f921'])
# model_vs_model_r2_graph_cv=px.pie(model_comparation_r2[model_comparation_r2['set']=='cv'],values='values',hole=.5,names='estimator',
#                                  title='Model Comparison in the Cv Set <br> <sub>by R2</sub>',
#                                  color_discrete_sequence=['#0d0887','#9c179e','#ed7953','#fb9f3a','#f0f921'])
# model_vs_model_r2_graph_test=px.pie(model_comparation_r2[model_comparation_r2['set']=='test'],values='values',hole=.5,names='estimator',
#                                  title='Model Comparison in the Test Set <br> <sub>by R2</sub>',
#                                  color_discrete_sequence=['#0d0887','#9c179e','#ed7953','#fb9f3a','#f0f921'])

# model_vs_model_r2_graph = px.bar(model_comparation_r2,x='set',y='values',color='estimator',barmode='group',text_auto='.2',
#                               title='R2 Models Comparation',labels={'estimator':'','value':'Mean Absolute Error','set':''},
#                               color_discrete_sequence=['#0d0887','#9c179e','#ed7953','#fb9f3a','#f0f921'])

# model_vs_model_r2_graph.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

model_vs_model_r2_graph_train = ds.load_plot_json(path_figures,'model_vs_model_r2_graph_train')
model_vs_model_r2_graph_cv = ds.load_plot_json(path_figures,'model_vs_model_r2_graph_cv')
model_vs_model_r2_graph_test = ds.load_plot_json(path_figures,'model_vs_model_r2_graph_test')
model_vs_model_r2_graph = ds.load_plot_json(path_figures,'model_vs_model_r2_graph')
model_vs_model_mse_graph = ds.load_plot_json(path_figures,'model_vs_model_mse_graph')

# ENSEMBLE REGRESSOR ---------------------------------------------------------->

# from sklearn.ensemble import VotingRegressor

# print('Entering... Ensemble')

# ensemble_mode = 'design' # Work mode 'train' to train the model, 'design' to design the page layout with a trained model
# ensemble_save_model = True # Save the model

# if ensemble_mode == 'train':

#     ensemble_model = VotingRegressor(estimators=[('svm',svm_model),('tree', tree_model), ('poly', poly_model)])
#     ensemble_model.fit(X_train,y_train)
#     y_pred_train_ensemble = ensemble_model.predict(X_train)
#     y_pred_cv_ensemble = ensemble_model.predict(X_cv)
#     y_pred_test_ensemble = ensemble_model.predict(X_test)
    
#     if ensemble_save_model == True:
#         joblib.dump(ensemble_model,os.path.join(path_models,'ensemble.joblib'))

# elif ensemble_mode == 'design':
#     ensemble_model = joblib.load(os.path.join(path_models,'ensemble.joblib'))
#     y_pred_train_ensemble = ensemble_model.predict(X_train)
#     y_pred_cv_ensemble = ensemble_model.predict(X_cv)
#     y_pred_test_ensemble = ensemble_model.predict(X_test)


# ensemble_accuracy_train,ensemble_accuracy_cv,ensemble_accuracy_test,ensemble_ind_train,\
# ensemble_ind_cv,ensemble_ind_test,ensemble_bias_variance \
#      = ds.single_regression_model_evaluation(y_train,y_pred_train_ensemble,\
#                                         y_cv,y_pred_cv_ensemble,y_test,y_pred_test_ensemble,
#                                         color_line=['#fdca26','#0d0887'],color_bar=['#f0f921','#0d0887'])

# learning_curve = ds.learning_curve(ensemble_model, X_train, y_train, cv=10, n_jobs=-1, scoring='r2', 
#         train_sizes=np.linspace(0.1, 1.0, 20), transform_to_positive = False, y_range = None, random_state=42)[0]

# with open(os.path.join(path_models,'learning_curve.json'),'a',encoding="utf-8") as f:
#                 f.write(plotly.io.to_json(learning_curve))

# <----------------------------- Dash Layout --------------------------------> #

app = Dash(__name__,external_stylesheets=[dbc.themes.COSMO],title='Diabetes',update_title='',
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, \
                             initial-scale=1.0'}]) # SOLAR, LUX

server = app.server

draw_figure_buttons = {'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]}

app.layout = dbc.Container([ 

    dbc.Row(dbc.Col([html.H1('Diabetes DataFrame')],width=6,className="title")),
    dbc.Row(dbc.Col([html.H4('by Exlonk Gil')],width=12)),

    # <------------------------------DATA INSIGHTS --------------------------> #

    dbc.Row(dbc.Col([html.H2('Raw Data Insights')],width=12,    
    style={"padding-top":"1rem","padding-bottom":"1rem","textAlign":"center"})),

    dbc.Row(dbc.Col([html.P('This dataset shows the progress of diabetes \
    as a number. Because of this, it is a supervised regression problem and the \
    estimators used to model it can be very wide.')])),

    dbc.Row([dbc.Col([html.P([html.Span(['Note:'],style={'color':'blue'}),html.Span([' ']),' All \
    figures are interactive, on some of them it can be drawn.'])])]),

    dbc.Row(dbc.Col([table],width=12)),

    dbc.Row(dbc.Col(html.Br(),width=12)), # LINEA EN BLANCO

    dbc.Row(dbc.Col([html.P("It can be seen for the values represented \
        in the table that the original data was delivered in a scaled form. \
        The \"shape\" and \"data type\" figures shows the amount of data that it is \
        treated and the categorical and numerical data in the set. It appears that \
        there is no categorical data, but this is because of the preprocessing \
        prior procedure, since the categorical 'sex' feature takes two nominal values.")])),
    
    dbc.Row(dbc.Col(html.Br(),width=12)), # LINEA EN BLANCO

    dbc.Row([dbc.Col([dcc.Graph(figure=shape)],width=5),dbc.Col
           ([dcc.Graph(figure=info_fig)],width=7)]),

    dbc.Row(dbc.Col([html.H2('Data Cleaning')],width=12,
    className="title",style={"textAlign": "center"})),
    
    # Duplicates and missing values

    dbc.Row(dbc.Col([html.H3('Duplicates and Missing Values')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('This section shows if there is some duplicated\
                              rows in the dataframe ' 
                              ' and the number of missing data per feature')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=duplicates)],width=3),\
             dbc.Col([dcc.Graph(figure=missing_values)],width=9)]),
    
    dbc.Row(dbc.Col([html.P('Like the figures shows, there is no missing or \
    duplicated data in the set.')])),

    # HISTOGRAMS

    dbc.Row(dbc.Col([html.H3('Histograms')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('For many machine learning models, it is \
    important to check the distribution type on the data. Different \
    histograms will be displayed for this purpose.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=histograms[0])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[1])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[2])],width=4)]),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=histograms[3])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[4])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[5])],width=4)]),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=histograms[6])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[7])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[8])],width=4)]),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=histograms[9])],width=6),\
             dbc.Col([dcc.Graph(figure=histograms[10])],width=6)]),
             
    dbc.Row(dbc.Col([html.P('Many of these figures look like have normal \
        distribution but when the Shapiro-Wilk and Kolmogorov-Smirnov test values \
        are analyzed it can be encountered that the data is not normalized \
        because these values are less than 0.05.')])),

    # <---------------------------DATA PREPARATION --------------------------> #

    # Outlier identification

    dbc.Row(dbc.Col([html.H3('Outlier Identification')],width=12,
                            className="subtitle", style={"textAlign": "left"})),
    
    dbc.Row(dbc.Col([html.P('Since this is a purely visual aid, a rescaling of \
     all features is used (again), this rescaling (called Robust Scaler) is \
     robust to outliers, this was done to take into account the target')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=box_plot_pure,config=draw_figure_buttons)],width=12)),

    dbc.Row(dbc.Col([dcc.Graph(figure=box_plot_lof,config=draw_figure_buttons)],width=12)),

    dbc.Row(dbc.Col([dcc.Graph(figure=box_plot_ifo,config=draw_figure_buttons)],width=12)),

    dbc.Row(dbc.Col([html.P("It can be seen of the three different forms of outlier \
    identification, that depending on the model, the amount of outlier data is widely \
    different, because of this all of them must be proven.")])),
    
    
    # <------------------------- FEATURE SELECTION -------------------------> #

    dbc.Row(dbc.Col([html.H2('Feature Selection')],width=12,
    className="title",style={"textAlign": "center"})),   

    dbc.Row(dbc.Col([html.H3('Statistics Feature Selection')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('This section shows some correlation metrics taking \
                              into account the nature of the predictors, \
                             that is, if they are categorical or numerical.')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=pearson_correlation)],width=12)),

    dbc.Row(dbc.Col([html.P('The Pearson correlation is used to encounter some\
    linear correlation within the analyzed features, in this case it can be denoted\
    that there are some linear relationship within the "target", the "body mass index"\
    and "the possibly log of serum", but it is very low to be conclusive for stick to\
    linear models. The only one linear relationship that is nearly acceptable is\
    between the "low-density lipoproteins" and the "total serum cholesterol",\
    it can be used to reduce the number of predictors.')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=spearman_correlation)],width=12)),

    dbc.Row(dbc.Col([html.P("While Pearson's correlation assesses linear \
    relationships, Spearman's correlation assesses monotonic relationships \
    (whether linear or not). The graph shows that the values are the same as\
    in the Pearson's case, this is because the Spearman correlation is based\
    in the Pearson correlation, and it implies that the relationship is a little\
    linear, but inconclusive.")])),

    dbc.Row(dbc.Col([dcc.Graph(figure=mutual_info_correlation)],width=12)),

    dbc.Row(dbc.Col([html.P('The mutual information regression is a method that \
    allows to encounter relationship between the numerical target and a numerical \
    or categorical feature, the relationship measure between two random variables is \
    a non-negative value, which measures the dependency between the variables. \
    It is equal to zero if and only if two random variables are independent, and \
    higher values mean higher dependency. In this scenario it exhibits that the "body mass index" \
    holds the most important relationship whiting the target, and that the "age",\
    "sex" and "total serum cholesterol" are the less important relationships.')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=aic_correlation)],width=12)),

    dbc.Row(dbc.Col([html.P('In the 1970s, Hirotugu Akaike, developed a metric \
    called AIC (Akaike’s Information Criteria) that penalizes adding terms to a model. \
    The goal is to find the model that minimizes AIC. In this case, the patron exhibited \
    by the other correlation methods keeps, with the "body mass index", "possibly log of \
    serum" and "average blood pressure" like the most important features.')])),

    dbc.Row(dbc.Col([html.P('For the feature selection and data preprocessing many \
    combinations of transformations were made, the linear regression model was \
    used as estimator (this because of the limited resources).  Between mutual \
    info regression, linear discriminant analysis, PCA, r regression and chi2 \
    analysis, the best feature selection was done  by r regression for the \
    numerical features and chi2 for the categorical ones. Between RobustScaler, \
    MinMaxScaler, MaxAbsScaler and StandardScaler, the MinMaxScaler showed \
    the best performance.')])),

    dbc.Row(dbc.Col([html.P('At the final stage, the pipeline selected for the \
        data preprocessing step is: ')])),

    dbc.Row(dbc.Col([html.P('OneHotEncoder for the categorical data (the "sex" feature), \
    LocalOutlierFactor for the outlier data, SelectKBest(score_func=r_regression, k=7) \
    for the numerical data and MinMaxScaler for rescale the data (again). ')])),

        
# <------------------------- MACHINE LEARNING MODELS -------------------------> #

    dbc.Row(dbc.Col([html.H2('Machine Learning Models')],width=12,
                         className="title",style={"textAlign": "center"})), 
    
    # SAMPLE DISTRIBUTION 

    dbc.Row(dbc.Col([html.H3('Sample Distribution')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The sample distribution graph is useful to check \
    the homogeneity of the sampling procedure. In this case, it looks well.')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=sample_distribution_fig)],width=12)),

    dbc.Row(dbc.Col([html.P('The split made for this set was 0.6 for the \
    training set, 0.2  for the cv (validation set) and 0.2 for test set.')])),

    # Baseline Model

    dbc.Row(dbc.Col([html.H3('Baseline Model')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('A baseline model is the simplest model that \
        produces answers that we can compare against. More complex models should do \
        better than baseline models and therefore gives numerical comparison. \
        The simplest model for a regression problem is to take the mean value of \
        the target and use it as a constant for any instance.')])),
    
    dbc.Row(dbc.Col([html.P('The following graphs illustrate the accuracy of the model in \
        the training and validation set, the "accuracy graph" is a figure with adjusted \
        curves made applying a "lowess" adjustment, meanwhile the "Individual error" \
        is a graph that shows the absolute error for each target data, whose size and \
        color is an indicator of error size.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_train,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_baseline_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=ind_baseline_cv,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_test,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_baseline_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('As expected, there is very little adjustment of this \
        approximation to the data. The R-squared is zero, and the mean absolute error \
        is greater than fifty, which is very high for data with a range between thirty \
        and three hundred twenty.')])),

    # Polynomial Regression --------------------------------------------------->

    dbc.Row(dbc.Col([html.H3('Polynomial Regression')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The first model used is the polynomial regression \
        simpler model, ten degrees are used to find the best fitting, \
        the vertical dashed line is used to mark the lowest train/cv gap value.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=mse_train_cv_poly_graph,config=draw_figure_buttons)],width=8),
             dbc.Col([dcc.Graph(figure=r2_train_cv_poly_graph)],width=4)]),

    dbc.Row(dbc.Col([html.P('The best degree (with the best train/cv gap) of the model is two, but it has low accuracy,\
        higher degrees improve the fitting in the training set, but down the fitting \
        in the validation set. More detailed graphs are shown:')])),
    
    
    dbc.Row([dbc.Col([dcc.Graph(figure=linear_regression_accuracy_train,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_linear_regression_accuracy_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_regression_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=ind_linear_regression_accuracy_cv,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_regression_accuracy_test,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_linear_regression_accuracy_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_regression_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('There is a good accuracy trade-off between the test \
        and the cross validation set, but in the test set the accuracy is very low, \
        this can be due to the size of the data set, the lack of correlation between \
        the predictors and the target and of course by the model used.')])),

    dbc.Row(dbc.Col([html.P('Different linear models with regularization were proven,\
        the final election was the Ridge Model with alpha equal to twenty-seven, this model,\
        like the simple linear one has  also low accuracy.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=mse_train_cv_rpoly_graph,config=draw_figure_buttons)],width=8),
             dbc.Col([dcc.Graph(figure=r2_train_cv_rpoly_graph)],width=4)]),

    # Model Evalutaion

    dbc.Row(dbc.Col([html.P('The next graphs will show the application of the \
     ten degree ridge model in the train, validation and test sets.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_train,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_poly_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=ind_poly_cv,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_test,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=ind_poly_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('The use of polynomial features within the Ridge \
    model improve the accuracy in the test set, but down the accuracy in the \
    training and cv sets. In conclusion, it is appropriate to add new features \
    to the models implemented, but its accuracy is low.')])),

    dbc.Row(dbc.Col([html.P('The next figure is a learning curve graph, \
    used to have an idea about the effect of the set size on the accuracy')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=learning_curve)],width=12)]),

    dbc.Row(dbc.Col([html.P("It can be seen from the graph that at the final stage \
    the size of the set appears to have a middle effect on the train/cv set gap. \
    Maybe this is because of the lack of correlation between the features and the target.")])),

    # KNeigh Neighbors -------------------------------------------------------->

    dbc.Row(dbc.Col([html.H3('KNeighbor')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The KNeighbors Regressor was the second estimator \
                             that was proven. Below are some validation curves \
                             that illustrate the impact of one hyperparameter or \
                             another on the accuracy metric.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure= knn_val_curve_1)],width=6),dbc.Col([dcc.Graph(figure=knn_val_curve_2)],width=6)]),
   
    dbc.Row([dbc.Col([dcc.Graph(figure= knn_val_curve_3)],width=4),dbc.Col([dcc.Graph(figure= knn_val_curve_4)],width=4),
            dbc.Col([dcc.Graph(figure= knn_val_curve_5)],width=4)]),


    dbc.Row(dbc.Col([html.P('In this case, the validation curve was made for some  \
        hyperparameters. According to the figures, the number of neighbors has a \
        significant effect on the train/CV gap, whereas the other factors have a small impact.')])),
    
    # Model Evaluation

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_train)],width=6),
             dbc.Col([dcc.Graph(figure=knn_ind_train,config=draw_figure_buttons)],width=6)]),


    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=knn_ind_cv,config=draw_figure_buttons)],width=6)]),
 
    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_test)],width=6),
             dbc.Col([dcc.Graph(figure=knn_ind_test,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('The model was tuned for the parameters shown in the \
        validation curves by a grid search, the best fit gives poor accuracy, even \
        below for the values that the polynomial model produces, though in the test \
        set shows a little improve. The absolute error is broad, even in the train set, \
        in conclusion, this model does not show any performance improvement over the linear model')])),

    # Support Vector Machine -------------------------------------------------->

    dbc.Row(dbc.Col([html.H3('Support Vector Machine')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('Another estimator used for modeling this problem is \
        the Support Vector Machine model. It has some hyperparameters for tuning, \
        like the kernel, gamma, and degree. Among them, the kernel has the biggest \
        impact on the train/CV gap and accuracy.')])),
    
    dbc.Row([dbc.Col([dcc.Graph(figure= svm_val_curve_1)],width=6),dbc.Col([dcc.Graph(figure=svm_val_curve_2)],width=6)]),
   
    dbc.Row([dbc.Col([dcc.Graph(figure= svm_val_curve_3)],width=6),dbc.Col([dcc.Graph(figure= svm_val_curve_4)],
            width=6)]),

    # Evaluation 

    dbc.Row([dbc.Col([dcc.Graph(figure=svm_accuracy_train)],width=6),
             dbc.Col([dcc.Graph(figure=svm_ind_train,config=draw_figure_buttons)],width=6)]),


    dbc.Row([dbc.Col([dcc.Graph(figure=svm_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=svm_ind_cv,config=draw_figure_buttons)],width=6)]),
 
    dbc.Row([dbc.Col([dcc.Graph(figure=svm_accuracy_test)],width=6),
             dbc.Col([dcc.Graph(figure=svm_ind_test,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=svm_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('This model illustrates some improvement in accuracy \
        at the test and cross validation sets, but the absolute error still remains big.')])),

    # Ensemble Tree

    dbc.Row(dbc.Col([html.H3('Ensemble Tree')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P("The ensemble tree (XGBoost) estimator's validation \
        curves indicate that the hyperparameters eta and max deep have a large \
        impact on the train/CV gap, but neither of all the hyperparameters shown \
        any improvement in the accuracy of the model without overfitting the data")])),
    
    dbc.Row([dbc.Col([dcc.Graph(figure= tree_val_curve_1)],width=6),dbc.Col([dcc.Graph(figure=tree_val_curve_2)],width=6)]),
   
    dbc.Row([dbc.Col([dcc.Graph(figure= tree_val_curve_3)],width=6),dbc.Col([dcc.Graph(figure= tree_val_curve_4)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure= tree_val_curve_5)],width=6),dbc.Col([dcc.Graph(figure= tree_val_curve_6)],width=6)]),

    # Model Evaluation

    dbc.Row([dbc.Col([dcc.Graph(figure=tree_accuracy_train)],width=6),
             dbc.Col([dcc.Graph(figure=tree_ind_train,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=tree_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=tree_ind_cv,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=tree_accuracy_test)],width=6),
             dbc.Col([dcc.Graph(figure=tree_ind_test,config=draw_figure_buttons)],width=6),
             dbc.Col([dcc.Graph(figure=tree_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P("This model has an improvement in accuracy on the \
    test set compared to previous models, despite this, the accuracy in \
    general is very low.")])),

    # Neural Network

    dbc.Row(dbc.Col([html.H3('Neural Network')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The next graph is a representation of the neural \
        network model used, it shows the number and relative size of the layers \
        as the relative number of connections')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=neural_network_graph)],width=12)),

    # Model Evaluation

    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_train)],width=6),
             dbc.Col([dcc.Graph(figure=network_ind_train,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_cv)],width=6),
             dbc.Col([dcc.Graph(figure=network_ind_cv,config=draw_figure_buttons)],width=6)]),
 
    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_test)],width=6),
             dbc.Col([dcc.Graph(figure=network_ind_test,config=draw_figure_buttons)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=network_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P("The neural model has some l2 and drop regularization, \
    multiple architectures were proven as well as multiple values for regularization, \
    but the accuracy doesn't improve nicely, despite this, it has the best \
    accuracy in the development set.")])),

    # MODEL COMPARISON

    dbc.Row(dbc.Col([html.H3('Model Comparison')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('This section shows a model comparison graph by \
    R2 or by mean absolute error (MAE).')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=model_vs_model_r2_graph_train)],width=4),
             dbc.Col([dcc.Graph(figure=model_vs_model_r2_graph_cv)],width=4),
             dbc.Col([dcc.Graph(figure=model_vs_model_r2_graph_test)],width=4)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=model_vs_model_mse_graph)],width=12)]),

    dbc.Row(dbc.Col([html.P("The best model in this case is the neural model, \
    followed by the svm and the polynomial one. Despite this, the accuracy of \
    any model on the sets is low, but in comparison with the baseline there is \
    an improvement of 30% approx. This lack of accuracy can be due mainly to the \
    low correlation between the features and the target or the size of the set.")])),

    # Ensemble Model

    # dbc.Row(dbc.Col([html.H3('Ensemble Model')],width=12,
    #                         className="subtitle",style={"textAlign": "left"})),

    # dbc.Row(dbc.Col([html.P('This section shows some correlation metrics taking \
    #                           into account the nature of the predictors, \
    #                          that is, if they are categorical or numerical.')])),

    # dbc.Row([dbc.Col([dcc.Graph(figure=ensemble_accuracy_train)],width=6),
    #          dbc.Col([dcc.Graph(figure=ensemble_ind_train,config=draw_figure_buttons)],width=6)]),

    # dbc.Row([dbc.Col([dcc.Graph(figure=ensemble_accuracy_cv)],width=6),
    #          dbc.Col([dcc.Graph(figure=ensemble_ind_cv,config=draw_figure_buttons)],width=6)]),
 
    # dbc.Row([dbc.Col([dcc.Graph(figure=ensemble_accuracy_test)],width=6),
    #          dbc.Col([dcc.Graph(figure=ensemble_ind_test,config=draw_figure_buttons)],width=6)]),

    # dbc.Row([dbc.Col([dcc.Graph(figure=ensemble_bias_variance)],width=12)]),

    # dbc.Row([dbc.Col([dcc.Graph(figure=learning_curve)],width=12)]),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Cierre del layout
    ],className="container")

# Dash Callbacks #

# @app.callback(
#    Output(table,component_property='children'),
#    Input(table_dropdown,component_property='value')
#   )

if __name__ == '__main__':
      app.run_server(port=8070)
