import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#load data
data = pd.read_csv(r"/home/lero/idrive/cmac/DDMAP/stability_studies_master/stability_dataset_manual_edit.csv")
data = data.drop(data.columns[20:32], axis=1)
data.drop(['Unnamed: 0.6', 'Unnamed: 0.5', 'Unnamed: 0.4', 'Unnamed: 0.7'], axis=1, inplace=True)

#Store values for API/ polymer, condition
original_api = data['API']
original_polymer = data['Polymer']
original_condition = data['condition']

#fill pure api with 0 for polymer mol desc
pure = data['Polymer']=='Pure'
polymer_descriptors = data.columns[233:]
data.loc[pure, polymer_descriptors] = 0
suffix = '_polymer'
data.rename(columns={col: suffix+col for col in polymer_descriptors}, inplace=True)

#drop conditions since these have been split into temp/ humidity
data.drop(['condition'], axis=1, inplace=True)

#fill na values with average
data.fillna(data.mean(numeric_only=True), inplace=True)

#drop columns with a mean value of 0
print('original DataFrame shape\n', data.shape)
numeric_means=data.mean(numeric_only=True)
mean_of_0 = numeric_means[numeric_means==0].index
data.drop(mean_of_0, inplace=True, axis=1)

data.drop(['API', 'Polymer', 'Average days stable'], inplace=True, axis=1)

df = pd.DataFrame(data.columns[:12])
df.rename(columns={0: 'Features'}, inplace=True)
print('\nAPI and Polymer features\n\n', df)
print('\n\nShape of dataframe after removal of mean = 0\n', data.shape)

#Define Features for ColumnTransformer (AFTER ALL DROPS within dataframe) ---
categorical_features = ['API', 'Polymer']
# Identify numerical features:
dont_scale_features = data.drop(['days_stable_min', 'GFA', 'is_pure', ], axis=1).columns.tolist()
numerical_features = [col for col in dont_scale_features if col not in categorical_features]

#split data
X = data.drop(['days_stable_min'], axis=1)
y_non_binary = data['days_stable_min']
y = (y_non_binary>=90).astype(int)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier

models = {
    # 'Logistic Regression': (
    #     LogisticRegression(max_iter=1000000),
    #     {
    #         'model__C': np.logspace(-4, 4, 20),
    #         'model__penalty': ['elasticnet'],
    #         'model__solver': ['saga'],
    #         'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    #     }
    # ),
    # 'SVC': (
    #     SVC(max_iter=10000000, random_state=42),
    #     {
    #         'model__penalty':['l2', 'l1'],
    #         'model__C': [0.01, 0.1, 1, 10, 100]  
    #     }
    # ),
    # 'K Neighbors Classifier': (
    #     KNeighborsClassifier(), 
    #     {
    #         'model__n_neighbors': np.arange(2,30,1)
    #     }
    # ),
    # 'catboost':(
    #     CatBoostClassifier(early_stopping_rounds=100),
    #     {
    #         'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
    #         'model__depth':[3, 10, 50],
    #         'model__iterations': [1000, 1500, 2000],
    #         'model__l2_leaf_reg': [1, 3, 5, 7, 10],
    #         'model__subsample': [0.5, 0.7, 0.9, 1]
    #     }
    # ),
    # 'Random Forest Classifier': (
    #     RandomForestClassifier(random_state=42), 
    #     {
    #         'model__n_estimators': [300, 500, 1000, 1500],
    #         'model__max_features': ['sqrt', 'log2', None],
    #         'model__max_depth': [None, 1, 2, 3, 5, 10],
    #         'model__min_samples_split': [2, 5, 10, 20],
    #         'model__max_samples': [0.5, 0.7, 0.9, 1]
    #     }
    # ),
    'XGBoost classifier': (
    XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        device='cuda',
        tree_method='hist'
    ),
        {
            'model__booster': ['gbtree'],
            'model__max_depth': [3, 6, 10],
            'model__subsample': [0.5, 0.8, 1],
            'model__colsample_bytree': [0.7, 0.9, 1],
            'model__learning_rate': [0.01, 0.1],
            'model__n_estimators': [300, 500, 800],
            'model__reg_alpha': [1e-2, 0.1, 1, 10],
            'model__reg_lambda': [1e-2, 0.1, 1, 10],
            #'feature_selection__max_features': [20, 30, 50]
        },
    ),
#     'MLP Classifier': (
#         MLPClassifier(max_iter=100000000, early_stopping=True), 
#         {
#             'model__hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 200)],
#             'model__activation': ['tanh', 'relu'],
#             'model__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
#             'model__solver': ['sgd', 'adam'],
#             'model__learning_rate': ['adaptive'],
#             'model__learning_rate_init': [0.001, 0.01],
#         }
#     )
}

#pre-processor for one-hot encoding and scaling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', StandardScaler(), numerical_features),
    ],
    remainder = 'passthrough' # Keep any other columns not explicitly transformed (e.g., if there are any not in num or cat)
)

#Nested cv
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
import pickle
from sklearn.metrics import f1_score
import os
from tqdm.auto import tqdm
import shutil

scorer = {
    'F1_score': make_scorer(f1_score, average='binary'),
    'Accuracy': 'accuracy',
    'ROC-AUC': 'roc_auc'
}

#groups for GroupKFold
groups = (original_api.astype(str)).values

#GroupKFold for outer cv
outer_cv = GroupKFold(n_splits=5) #change n_splits to 80:20
inner_cv = GroupKFold(n_splits=5) #change n_splits to 80:20

#directory to save the models
save_directory = '/home/lero/idrive/cmac/DDMAP/Stability studies/Model_results/Apr26_re_train/Classifier/Default'
os.makedirs(save_directory, exist_ok=True)
# Save a copy of the running script
shutil.copy(__file__, os.path.join(save_directory, "XGBrun_script_backup.py"))

results = {}

for model_name, (classifier, param_grid) in tqdm(models.items(), desc='models', total=len(models)):
    print('Model:', model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', classifier)
    ])
    
  
    # Perform nested cross-validation
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring=scorer, refit='F1_score', verbose=10, n_jobs=1)
    
    fit_params = {'groups': groups}
    
    # Evaluate outer loop scores
    nested_score = cross_validate(grid_search, X, y, groups=groups, cv=outer_cv, params=fit_params, scoring=scorer, n_jobs=1, return_train_score=False)
    
    # Get predictions
    predictions = cross_val_predict(grid_search, X, y, groups=groups, cv=outer_cv, params=fit_params, method='predict', n_jobs=1)

    # Fit to find best parameters
    grid_search.fit(X, y, **fit_params)
    best_params = grid_search.best_params_
    
    # Save the best model
    best_model = grid_search.best_estimator_
    model_file_path = os.path.join(save_directory, f'{model_name}_best_model.pkl')
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    results[model_name] = {
        'nested_score': nested_score,
        'ground_truth': y.values,
        'predictions': predictions,
        'best_params': best_params,
    }
  
dictionary_file_path = os.path.join(save_directory, 'Default_Classifiers_results_dictionary.pkl')
with open(dictionary_file_path, 'wb') as f:
    pickle.dump(results, f)

print('Finito')