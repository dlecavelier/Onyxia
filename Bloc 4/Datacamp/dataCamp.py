import pandas as pd

pd.set_option('display.max_columns', None)

y_df = pd.read_csv(r"C:\Users\cepe-s3-02\Desktop\David\Bloc 4\DataCamp\engieY.csv", sep=";")
x_df = pd.read_csv(r"C:\Users\cepe-s3-02\Desktop\David\Bloc 4\DataCamp\engieX.csv", sep=";")

# Suppression des lignes avec des valeurs manquantes
y_df = y_df.dropna()
x_df = x_df.dropna()

# Jointure à gauche (left join) sur une colonne commune, par exemple 'id'
# Remplacez 'id' par le nom de la colonne clé commune à vos deux DataFrames
merged_df = x_df.merge(y_df, how='left', on='ID')

# Conserver uniquement les lignes où mac_code vaut 'wt1'
if 'MAC_CODE' in merged_df.columns:
    merged_df = merged_df[merged_df['MAC_CODE'] == 'WT1']
    print("Lignes où MAC_CODE == 'WT1' :")
    print(merged_df)
else:
    print("La colonne 'MAC_CODE' n'existe pas dans merged_df.")

# Conserver uniquement une ligne sur 6 selon l'ordre de la colonne 'DATE_TIME'
if 'Date_time' in merged_df.columns:
    merged_df = merged_df.sort_values('Date_time')
    merged_df = merged_df.iloc[::6]
    print("DataFrame avec une ligne sur 6 selon Date_time :")
    print(merged_df)
else:
    print("La colonne 'Date_time' n'existe pas dans merged_df.")

# Supprimer la colonne 'MAC_CODE' et toutes les colonnes se terminant par _min, _max, _std
cols_to_drop = ['MAC_CODE', 'ID', 'Date_time', 'Absolute_wind_direction', 'Nacelle_angle'] + [col for col in merged_df.columns if col.endswith(('_min', '_max', '_std'))]
merged_df = merged_df.drop(columns=cols_to_drop)

merged_df = merged_df.reset_index(drop=True)

# Enregistrement du DataFrame trié
merged_df.to_csv(
    r"C:\Users\cepe-s3-02\Desktop\David\Bloc 4\DataCamp\merged_df_trie.csv",
    index=False,
    sep=';',
    decimal=','
)

import plotly.express as px

# Exemple : tracer un nuage de points (scatter) avec plotly (zoom possible)
fig = px.scatter(merged_df, x='Rotor_speed', y='TARGET')  # points non reliés
fig.update_layout(title="Nuage de points interactif avec zoom")
#fig.show()

features = merged_df.drop(columns=['TARGET'])
label = merged_df['TARGET']

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(features, label, test_size= 0.1, random_state= 18)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= 0.1, random_state= 42)

print(len(features))      # total initial
print(len(X_train_val))   # après premier split
print(len(X_val))         # après deuxième split
print(len(X_test))        # test set

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_val)
print(y_predict)
y_predict = lr.predict(X_val)
print(y_predict)
mse = mean_squared_error(y_val, y_predict)
mae = mean_absolute_error(y_val, y_predict)
r2 = r2_score(y_val, y_predict)
print(f"Mean Squared Error (MSE) : {mse}")
print(f"Mean Absolute Error (MAE) : {mae}")
print(f"R2 Score : {r2}")

# Prédire sur X_test et mesurer la performance avec y_test
y_test_predict = lr.predict(X_test)

mse_test = mean_squared_error(y_test, y_test_predict)
mae_test = mean_absolute_error(y_test, y_test_predict)
r2_test = r2_score(y_test, y_test_predict)

print(f"Test Mean Squared Error (MSE) : {mse_test}")
print(f"Test Mean Absolute Error (MAE) : {mae_test}")
print(f"Test R2 Score : {r2_test}")

print(f"Nombre de lignes de y_test_predict : {len(y_test_predict)}")

import plotly.graph_objects as go
# Tracer la régression linéaire et les erreurs sans relier les points
fig = go.Figure()

# Points réels
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test,
    mode='markers',
    name='Valeurs réelles'
))

# Prédictions du modèle (points non reliés)
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test_predict,
    mode='markers',
    name='Prédictions'
))

# Erreurs (résidus)
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test - y_test_predict,
    mode='markers',
    name='Erreurs (résidus)'
))

fig.update_layout(
    title="Régression linéaire et erreurs (points non reliés)",
    xaxis_title="Index",
    yaxis_title="Valeur",
    legend_title="Légende"
)

#fig.show()

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state= 42)

rf.fit(X_train, y_train)
# Prédictions du Random Forest sur X_val
y_rf_predict = rf.predict(X_val)

# Calcul et affichage des métriques pour le Random Forest
mse_rf = mean_squared_error(y_val, y_rf_predict)
mae_rf = mean_absolute_error(y_val, y_rf_predict)
r2_rf = r2_score(y_val, y_rf_predict)

print(f"Random Forest Mean Squared Error (MSE) : {mse_rf}")
print(f"Random Forest Mean Absolute Error (MAE) : {mae_rf}")
print(f"Random Forest R2 Score : {r2_rf}")

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(
    n_splits= 5,
    shuffle=True, # les données sont mélangées aléatoirement avant d’être découpées en folds.
    random_state=42, 
    
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

metric_grid = ['accuracy', 'f1', 'roc_auc']

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42, max_iter= 2000)

from sklearn.svm import SVC

svm = SVC()

param_rf = {
    'classifier__n_estimators' : [100, 200, 500],
    'classifier__max_depth' : [10, 20, None],
}

num_transformer = Pipeline(steps= [
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps= [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown= 'ignore'))
])

from sklearn.compose import ColumnTransformer

num_features = features.select_dtypes(include= ['int32', 'float64']).columns
cat_features = features.select_dtypes(include= ['object']).columns

preprocessor = ColumnTransformer(transformers= [
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

pip_rf = Pipeline(steps=[
    ('preproc', preprocessor),
    ('classifier', rf)
]
)

grid_rf = GridSearchCV(
    estimator= pip_rf,
    param_grid= param_rf,
    cv = cv,
    scoring = 'r2', # metric selon lequel on détermine le best_result,
    refit = True # par defaut
)

print(grid_rf.best_params_)
print(grid_rf.best_score_)
print(grid_rf.score(X_test, y_test))

param_knn = {
    'preproc__num__scaler' : [StandardScaler(), MinMaxScaler(), Normalizer()],
    'classifier__n_neighbors' : [5, 7, 10,],
    'classifier__weights' : ['uniform', 'distance'],
    'classifier__metric' : ['euclidean', 'manhattan']
}

pip_knn = Pipeline(steps=[
    ('preproc' , preprocessor),
    ('classifier' , knn)
    
])

grid_knn = GridSearchCV(
    estimator=pip_knn,
    param_grid=param_knn,
    cv = cv,
    scoring = 'r2',
    refit = True
)

param_gb = {
    'classifier__n_estimators' : [200, 500, ],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth' : [7, 10]
}

pip_gb = Pipeline(steps = [
    ('preproc', preprocessor),
    ('classifier', gb)
])

grid_gb = GridSearchCV(
    estimator=pip_gb,
    cv = cv,
    param_grid=param_gb,
    scoring = 'r2'
)

pip_mlp = Pipeline(
    steps=[
        ('preproc', preprocessor),
        ('classifier', mlp)
    ]
)

param_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__solver': ['adam', 'sgd'], # l'algorithme d'optimisation utilisé pour mettre à jour les poids lors de l'entraînement du réseau de neurones
    'classifier__alpha': [0.0001, 0.001], # dans sklearn, régularisation est toujours ridge, alpha correspond à l'intensité de la pénalité
}

grid_mlp = GridSearchCV(
    estimator= pip_mlp,
    param_grid= param_mlp,
    cv = cv,
    scoring= 'r2'
)

pip_svm = Pipeline(steps=[
    ('preproc', preprocessor),
    ('classifier', svm)
])

param_svm = {
    'classifier__kernel' : ['linear', 'rbf', 'poly', 'sigmoid'],
    'classifier__C' : [0.01, 0.1, 1, 10, 100, ], # Le paramètre C en SVM (sklearn.svm.SVC) est bien utilisé pour tous les noyaux, pas seulement pour le linéaire.
    # ici le C est différent dans l'ols, il s'agit C : poids qui équilibre la régularisation et les erreurs. Donc il y a toujours un C
    'classifier__degree' : [2, 3, 4]
}

models = [
    ('lr', lr, {'classifier__C': [1.0]} ), # inverse de lambda, λ=1 → régularisation "standard"
    ('rf', rf, param_rf),
    ('knn', knn, param_knn),
    ('gb', gb, param_gb),
    ('mlp', mlp, param_mlp),
    ('svm', svm, param_svm)
]

preproc_list = [
    {'id' : 'basic', 'object': preprocessor},
    #{'id' : 'interact', 'object': preproc_interact},
    #{'id' : 'spline', 'object': preproc_spline},
]

results = []
for preproc in preproc_list : 
    preproc_id = preproc['id']
    preproc_object = preproc['object']
    for model_id, model_object, param in models : 
        print(f'Model {model_id} with preprocessor {preproc_id}')
        pip = Pipeline(steps= [
            ('preproc', preproc_object),
            ('classifier', model_object)
        ])

        grid = GridSearchCV(
            estimator= pip,
            cv=cv, 
            param_grid = param,
            scoring= metric_grid,
            refit = 'r2'
        ).fit(X_train_val, y_train_val)

        results.append(
            {
                'preprocessor' : preproc_id,
                'model' : model_id,
                'best_param' : grid.best_params_,
                'best_score' : grid.best_score_,
                'final_prediction' : grid.score(X_test, y_test)
            }
        )

pd.DataFrame(results)