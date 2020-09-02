##################Libraries####################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import validation_curve
from IPython.core.display import HTML

#################Basic data format###################
path = "C:/Users/User/Desktop/players_20.csv"
sns.set(style='darkgrid')
data = pd.read_csv(path)
data.head()

############Feature engineering####################
#Remove useless columns and goalkeepers
data = data.iloc[:,:len(data.columns) - 26]
data = data.drop(columns = ["sofifa_id", "player_url", "body_type", "real_face", "release_clause_eur", "nation_position",
                     "player_tags","team_jersey_number", "loaned_from", "joined", "nation_jersey_number",
                    "player_traits","international_reputation"])
data = data[data.columns.drop(list(data.filter(regex= 'gk_')))]
data = data[data.columns.drop(list(data.filter(regex= 'goalkeeping_')))]
data = data.dropna(subset = ["team_position"])
data = data[~data.player_positions.isin(["GK"])]

#Select player only from the best leagues
premierLeague_teams = ["Liverpool", "Manchester City", "Leicester City","Chelsea","Manchester United","Wolverhampton Wanderers",
                       "Sheffield United","Tottenham Hotspur","Arsenal","Burnley","Crystal Palace","Everton",
                      "Newcastle United", "Southampton", "Brighton & Hove Albion","West Ham United", "Watford","Bournemouth",
                       "Aston Villa", "Norwich City"]
bundesliga_teams = ["FC Bayern München", "Borussia Dortmund", "Bayer 04 Leverkusen","RB Leipzig","Borussia Mönchengladbach","Eintracht Frankfurt",
                       "FC Schalke 04","SV Werder Bremen","TSG 1899 Hoffenheim","VfL Wolfsburg","Hertha BSC","1. FC Köln",
                      "FC Augsburg", "1. FSV Mainz 05", "SC Freiburg","Fortuna Düsseldorf", "1. FC Union Berlin","SC Paderborn 07"]

primeraDivision_teams = ["FC Barcelona", "Real Madrid", "Atlético Madrid","Valencia CF","Real Betis","Sevilla FC","Athletic Club de Bilbao",
                       "Real Sociedad","Villarreal CF","Getafe CF","Levante UD","RC Celta","RCD Espanyol","SD Eibar",
                         "Real Valladolid CF", "Deportivo Alavés","CD Leganés", "Granada CF","CA Osasuna","RCD Mallorca"]
serieA_teams = ["Juventus", "Inter", "Napoli","Lazio","Atalanta","Milan", "Roma", "Torino", "Fiorentina", "Brescia",
                       "Cagliari","Udinese","Genoa","Sampdoria","Parma","Sassuolo", "Bologna", "Hellas Verona", "SPAL", "Lecce"]
ligueOne_teams = ["Paris Saint-Germain", "Olympique Lyonnais", "AS Monaco","Olympique de Marseille","LOSC Lille","AS Saint-Étienne",
                       "Stade Rennais FC","FC Girondins de Bordeaux","Montpellier HSC","OGC Nice","FC Nantes","RC Strasbourg Alsace",
                      "Angers SCO", "Toulouse Football Club", "Amiens SC","Stade de Reims", "FC Metz","Stade Brestois 29",
                       "Dijon FCO", "Nîmes Olympique"]
data = data[data.club.isin(premierLeague_teams + bundesliga_teams + primeraDivision_teams + serieA_teams + leagueOne_teams)]

#Create colum with leagues
data["league"] = data[["club"]]
data[["league"]] = data.league.replace(premierLeague_teams, "Premier League")
data[["league"]] = data.league.replace(bundesliga_teams, "Bundesliga")
data[["league"]] = data.league.replace(primeraDivision_teams, "Primera Division")
data[["league"]] = data.league.replace(serieA_teams, "Serie A")
data[["league"]] = data.league.replace(leagueOne_teams, "Ligue 1")

#Get main player position
data["main_position"] = data.player_positions.str.split(',').str[0]

#Create Column with 3 general positions(Defender, Midfielder, Striker)
data["general_position"] = data[["main_position"]]
data[["general_position"]] = data.general_position.replace(["CB"], "Central Defender")
data[["general_position"]] = data.general_position.replace(["LB", "RB", "RWB", "LWB"], "Side Defender")
data[["general_position"]] = data.general_position.replace(["CM", "CDM", "CAM"], "Central Midfielder")
data[["general_position"]] = data.general_position.replace(["LM", "RM"], "Side Midfielder")
data[["general_position"]] = data.general_position.replace(["LW", "RW"], "Winger")
data[["general_position"]] = data.general_position.replace(["ST", "CF"], "Striker")

#Create column with number of player position
data["number_of_positions"] = [len(part) for part in data.player_positions.str.split(",")]

#Is there any observation with null cell
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()

############Visual analysis####################

#Countplot leagues
fig = plt.figure(figsize = (8,6))
chart = sns.countplot(x = 'league',
              data = data,
              order = data['league'].value_counts().index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set(xlabel='Liga', ylabel='Liczba piłkarzy')
plt.show()

#Countplot number of positions
fig = plt.figure(figsize = (8,6))
chart = sns.countplot(x = 'number_of_positions',
              data = data,
              order = data['number_of_positions'].value_counts().index)
chart.set(xlabel='Liczba pozycji', ylabel='Liczba piłkarzy')
plt.show()

#Countplot general positions
fig = plt.figure(figsize = (8,6))
chart = sns.countplot(x = 'general_position',
              data = data,
              order = data['general_position'].value_counts().index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set(xlabel='Pozycja', ylabel='Liczba piłkarzy')

plt.show()

#Barplot wages ~ general positions
fig = plt.figure(figsize = (8,6))
chart = sns.boxplot(data['general_position'],data['wage_eur'],showfliers=False)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
chart.set(xlabel='Pozycja', ylabel='Zarobki')
plt.show()

#Barplot wages ~ leagues and general positions
fig = plt.figure(figsize = (12,8))
chart = sns.boxplot(x="league", y="wage_eur", hue="general_position", data=data,showfliers=False) 
chart.set(xlabel='Liga', ylabel='Zarobki')
plt.show()

plt.clf()

#Boxplots depends on number of positions
fig = plt.figure(figsize = (8,6))
chart = sns.boxplot(x="number_of_positions", y="wage_eur", data=data,showfliers=False)
chart.set(xlabel='Liczba pozycji', ylabel='Zarobki')
plt.show()

plt.clf()

########################Determinants of footballers wages############################

########################Data preparation######################

#Remove useless columns and change general position to dummy variables
data_regression = data.drop(columns = ["short_name", "long_name", "dob", "height_cm", "weight_kg", "nationality",
                     "club", "preferred_foot", "work_rate","contract_valid_until","team_position", "player_positions",
                                      "main_position"])
dummy = pd.get_dummies(data_regression['general_position'])
data_regression = data_regression.merge(dummy, left_index = True, right_index = True)
data_regression = data_regression.drop(columns = ["general_position","Striker"])

#Select 20 most useful variables
def selecting_variables(data, league):
    data = data.loc[data['league'] == league]
    data = data.drop(columns = "league")
    data_regression_vars=data.columns.values.tolist()
    
    y=['wage_eur']
    X=[i for i in data_regression_vars if i not in y]

    lin_regression = LinearRegression()
    rfe = RFE(lin_regression, 18)
    rfe = rfe.fit(data.drop(columns = "wage_eur"), data['wage_eur'].values.ravel())
    result = [x for x, y in zip(X, rfe.support_) if y]
    data = data.loc[:,y + ['value_eur']+result ]
    
    return(data)

#Split the data set by league 
data_regression_premier = selecting_variables(data_regression, "Premier League")
data_regression_bundes = selecting_variables(data_regression, "Bundesliga")
data_regression_primera = selecting_variables(data_regression, "Primera Division")
data_regression_serie = selecting_variables(data_regression, "Serie A")
data_regression_ligue = selecting_variables(data_regression, "Ligue 1")

#Building optimal model based on Backward Elimination
def backward_elimination(data,sl):
    x = data.iloc[:,1:]
    x.insert(0, "const",1)
    y = data.iloc[:, 0]
    x_opt = x
    
    while sl != 0:
        regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
        if sl < regressor_OLS.pvalues.max():
            max_index = np.where(regressor_OLS.pvalues == regressor_OLS.pvalues.max())
            x_opt = x_opt.drop(x_opt.columns[max_index], axis=1)
        else:
            sl = 0
    return x_opt

x_opt_premier =backward_elimination(data_regression_premier,0.05)
x_opt_bundes = backward_elimination(data_regression_bundes,0.05)
x_opt_primera = backward_elimination(data_regression_primera,0.05)
x_opt_serie = backward_elimination(data_regression_serie,0.05)
x_opt_ligue = backward_elimination(data_regression_ligue,0.05)

########################Premier league - determinants######################

#Describe selected variables
data_regression_premier.drop(columns = ["Central Defender", "Central Midfielder", "Side Defender", "Side Midfielder",
                                       "Winger"]).describe()
    
#Correlation heatmap
def corrPlot(data):
    cols = data.columns.tolist()
    fig = plt.figure(figsize = (12,10))
    cm = np.corrcoef(data[cols].values.T)
    sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f',yticklabels = cols, xticklabels = cols)
    plt.show()
corrPlot(pd.concat([data_regression_premier.iloc[:,0], x_opt_premier.iloc[:,1:]], axis=1))

OLS(data_regression_premier.iloc[:, 0],x_opt_premier.iloc[:,[0,1,2,3,4,9]]).fit().summary()

########################Bundesliga - determinants######################

#Describe selected variables
data_regression_bundes.describe()

corrPlot(pd.concat([data_regression_bundes.iloc[:,0], x_opt_bundes.iloc[:,1:]], axis=1))

OLS(data_regression_bundes.iloc[:, 0],x_opt_bundes.iloc[:,[0,1,4,5,10]]).fit().summary()

########################Primera Division - determinants######################

#Describe selected variables
data_regression_primera.describe()

corrPlot(pd.concat([data_regression_primera.iloc[:,0], x_opt_primera.iloc[:,1:]], axis=1))

x_opt_primera.insert(0, "const",1)
OLS(data_regression_primera.iloc[:, 0],x_opt_primera.iloc[:,[0,1,2,8]]).fit().summary()

########################Serie A - determinants######################

#Describe selected variables
data_regression_serie.describe()

corrPlot(pd.concat([data_regression_serie.iloc[:,0], x_opt_serie.iloc[:,1:]], axis=1))

OLS(data_regression_serie.iloc[:, 0],x_opt_serie.iloc[:,[0,1,2,3,4]]).fit().summary()

########################Ligue 1 - determinants######################

#Describe selected variables
data_regression_ligue.describe()

corrPlot(pd.concat([data_regression_ligue.iloc[:,0], x_opt_ligue.iloc[:,1:]], axis=1))

OLS(data_regression_ligue.iloc[:, 0],x_opt_ligue.iloc[:,[0,1,2,5,6,7]]).fit().summary()

########################Football players classification######################

#Remove useless columns
data_classification = data.drop(columns = ["short_name", "long_name", "dob", "nationality", "club", "preferred_foot",
                                       "work_rate","contract_valid_until","team_position", "player_positions","main_position"])

#Split dataset
X = data_classification.drop(columns = ["general_position","league"])
y = data_classification['general_position']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Create frame with cross validation results
def cv_res_frame(gs, index , split = "__",split_str = 1):
    res_frame = pd.DataFrame([gs.best_params_], index = [index])
    res_frame.columns = res_frame.columns.str.split(split).str[split_str]
    return(pd.concat([pd.DataFrame({"Best mean score":[gs.best_score_]}, index = [index]),res_frame], axis = 1))

#Create heatmap for confusion matrices (train and test sets)
def conf_matrix_plot(y_train,y_test, y_pred_train, y_pred_test,labels):
    #Prepare frames with confusion matrices
    df_cm_train = pd.DataFrame(confusion_matrix(y_train, y_pred_train), index = labels,
                      columns = labels)
    df_cm_test = pd.DataFrame(confusion_matrix(y_test, y_pred_test), index = labels,
                      columns = labels)
    #Set plot options
    fig, ax =plt.subplots(1,2,figsize=(13,6))
    fig.tight_layout(pad=5.0)
    
    #Train set heatmap
    train_heatmap = sns.heatmap(df_cm_train, annot=True, fmt = "g", ax = ax[0])
    train_heatmap.set_yticklabels(train_heatmap.get_yticklabels(), rotation=35)
    train_heatmap.set_title('Train set confusion matrix')
    
    #Test set heatmap
    test_heatmap = sns.heatmap(df_cm_test, annot=True, fmt = "g",ax = ax[1])
    test_heatmap.set_yticklabels(test_heatmap.get_yticklabels(), rotation=35)
    test_heatmap.set_title('Test set confusion matrix')
    fig.show()

#Create frame with train and test predictions results
def train_test_res(gs, X_train, y_train, X_test,y_test):
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    return(pd.DataFrame({"Train score": [clf.score(X_train,y_train)], "Test score": [clf_knn.score(X_test,y_test)]},
                        index = ["Results"]))
    
########################KNN - results######################

#Combining standardisation with the model
pipe_knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier())
#List Hyperparameters that we want to tune.
leaf_size = list(range(25,35))
n_neighbors = list(range(1,30))
p=[1,2]

#Convert to dictionary
params_knn = {"kneighborsclassifier__leaf_size":leaf_size, "kneighborsclassifier__n_neighbors":n_neighbors,
              "kneighborsclassifier__p":p}

#Grid search
gs_knn = GridSearchCV(estimator = pipe_knn, param_grid = params_knn, scoring = 'accuracy', cv = 10)

gs_knn = gs_knn.fit(X_train, y_train)

#Results
cv_res_frame(gs_knn, "KNN CV result")

labels = set(data_classification['general_position'])
y_train_pred = gs_knn.predict(X_train)
y_test_pred = gs_knn.predict(X_test)
conf_matrix_plot(y_train,y_test,y_train_pred,y_test_pred ,labels)

train_test_res(gs_knn, X_train, y_train, X_test,y_test)

########################Decision tree- results######################

pipe_random_repair = make_pipeline(StandardScaler(),
            DecisionTreeClassifier(criterion = 'entropy',
                                   min_samples_leaf = 4,
                                   min_samples_split = 10,
                                   max_features = 'auto',
                                   random_state = 1))

param_range = range(5,20)
train_scores, test_scores =validation_curve(estimator = pipe_random_repair,
                                                         X = X_train,
                                                         y = y_train,
                                                         param_name = 'decisiontreeclassifier__max_depth',
                                                         cv = 5,
                                                         param_range = param_range)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores,axis =1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores,axis =1)

fig = plt.figure(figsize = (8,6))
plt.plot(param_range, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Dokladnosc uczenia')
plt.fill_between(param_range,train_mean + train_std, train_mean - train_std,
                 alpha = 0.15, color = 'blue')
plt.plot(param_range, test_mean,
         color = 'green', linestyle = '--', marker = 's',
         markersize = 5, label = 'Dokladnosc walidacji')
plt.fill_between(param_range,test_mean + test_std, test_mean - test_std,
                 alpha = 0.15, color = 'green')
plt.xlabel('Parametr max_depth')
plt.ylabel('Dokladnosc')
plt.legend(loc = 'lower right')
plt.ylim([0.5,1.03])
plt.show()

#Combining standardisation with the model
pipe_tree = make_pipeline(StandardScaler(),
                        DecisionTreeClassifier(random_state = 1))
#List Hyperparameters that we want to tune.
criterion = ['gini','entropy']
max_features = ['auto', 'sqrt']
max_depth = [4,5,6]
min_samples_split = [2, 5, 10,12,15]
min_samples_leaf = [1, 2, 4]
#Convert to dictionary
params_tree= {"decisiontreeclassifier__criterion":criterion, "decisiontreeclassifier__max_features":max_features,
              "decisiontreeclassifier__max_depth":max_depth, "decisiontreeclassifier__min_samples_split": min_samples_split,
              "decisiontreeclassifier__min_samples_leaf": min_samples_leaf}

#Grid search
gs_tree = GridSearchCV(estimator = pipe_tree, param_grid = params_tree, scoring = 'accuracy', cv = 10,return_train_score = True)

gs_tree.fit(X_train, y_train)

cv_res_frame(gs_tree,index = "Tree CV results")

labels = set(data_classification['general_position'])
y_train_pred = gs_tree.predict(X_train)
y_test_pred = gs_tree.predict(X_test)
conf_matrix_plot(y_train,y_test,y_train_pred,y_test_pred ,labels)

train_test_res(gs_tree, X_train, y_train, X_test,y_test)
    
########################Random Forest - results######################

pipe_random_repair = make_pipeline(StandardScaler(),
            RandomForestClassifier(max_features = 'sqrt',
                                   bootstrap = True,
                                   min_samples_leaf = 1,
                                   min_samples_split = 2,
                                   n_estimators = 400,
                                   random_state = 1))

from sklearn.model_selection import validation_curve

param_range = [3,4,5,6,7,8,9,10]
train_scores, test_scores =validation_curve(estimator = pipe_random_repair,
                                                         X = X_train,
                                                         y = y_train,
                                                         param_name = 'randomforestclassifier__max_depth',
                                                         cv = 5,
                                                         param_range = param_range)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores,axis =1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores,axis =1)

fig = plt.figure(figsize = (8,6))
plt.plot(param_range, train_mean,
         color = 'blue', marker = 'o',
         markersize = 5, label = 'Dokladnosc uczenia')
plt.fill_between(param_range,train_mean + train_std, train_mean - train_std,
                 alpha = 0.15, color = 'blue')
plt.plot(param_range, test_mean,
         color = 'green', linestyle = '--', marker = 's',
         markersize = 5, label = 'Dokladnosc walidacji')
plt.fill_between(param_range,test_mean + test_std, test_mean - test_std,
                 alpha = 0.15, color = 'green')
plt.xlabel('Parametr max_depth')
plt.ylabel('Dokladnosc')
plt.legend(loc = 'lower right')
plt.ylim([0.6,1.03])
plt.show()

#Combining standardisation with the model
pipe_random = make_pipeline(StandardScaler(),
                        RandomForestClassifier(random_state = 1))
#List Hyperparameters that we want to tune.
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 4)]
max_features = ['auto', 'sqrt']
max_depth = [4,5]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

#Convert to dictionary
params_random = {"randomforestclassifier__n_estimators":n_estimators, "randomforestclassifier__max_features":max_features,
              "randomforestclassifier__max_depth":max_depth, "randomforestclassifier__min_samples_split": min_samples_split,
              "randomforestclassifier__min_samples_leaf": min_samples_leaf,"randomforestclassifier__bootstrap": bootstrap}

#Grid search
gs_random = GridSearchCV(estimator = pipe_random, param_grid = params_random, scoring = 'accuracy', cv = 10,return_train_score = True)

gs_random = gs_random.fit(X_train, y_train)

labels = set(data_classification['general_position'])
y_train_pred = gs_random.predict(X_train)
y_test_pred = gs_random.predict(X_test)
conf_matrix_plot(y_train,y_test,y_train_pred,y_test_pred ,labels)

train_test_res(gs_random, X_train, y_train, X_test,y_test)
    
########################Summary######################

#Comparative chart    
fig = plt.figure(figsize = (8,6))
results = pd.DataFrame({"Result":[0.827179,0.810931, 0.749385,0.689808, 0.825702,0.799114],"Algorithm": ["KNN","KNN", "Tree", "Tree", "Random Forest", "Random Forest"],
                         "Dataset": ["train","test","train","test","train","test"]})

sns.barplot(x="Algorithm", y="Result", hue="Dataset", data=results)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()