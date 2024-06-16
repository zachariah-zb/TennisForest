import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#DATA CLEANING SECTION/
# create a dataframe
years = list(range(2021,2022)) #array of consecutive years - starts AT first number ends BEFORE second number
#reads all years in the above array and concatenates the datasets of each year
data = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_' + str(years[0]-1) + '.csv')
for year in years:
    url = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_' + str(year) + '.csv'
    #data = np.concatenate([data, pd.read_csv(url)]) this one is bad for some reason
    data = pd.concat([data, pd.read_csv(url)])


#simplifies dataset to useful variables
data_simplified = data[['winner_name', 'loser_name','tourney_name','surface','draw_size','tourney_level','winner_rank','winner_rank_points','loser_rank','loser_rank_points','round','loser_age','winner_age']]

#changes davis cup values in tourney_name to be just Davis Cup instead of, for example, Davis Cup WG SF: BEL vs AUS
def replace_davis_cup(text):
    return re.sub(r'\bDavis Cup\b.*', 'Davis Cup', text)
data_simplified.loc[:,'tourney_name'] = data_simplified['tourney_name'].apply(replace_davis_cup)

# Perform one-hot encoding for multiple categorical variables
data_simplified = pd.get_dummies(data_simplified, columns=['tourney_name', 'surface', 'tourney_level', 'round'], dtype=int)
#data_simplified = data_simplified.astype(int)

#Get just one players data
names = ['Alexander Zverev']
player_win_data = data_simplified[data_simplified['winner_name'].isin(names)]
player_lose_data = data_simplified[data_simplified['loser_name'].isin(names)]
player_data = pd.concat([player_win_data, player_lose_data])

#Adds a column variable 'match result' 1=win, 0=loss
player_data['match result'] = np.where(player_data['winner_name'] == names[0], 1, 0)
#novak_data['result'] = novak_data.winner_name.eq(names[0]).mul(1)
print(player_data)



#DECISION FOREST SECTION
#slice data into indepedent and dependent variables
variables_to_drop = ['match result','winner_name','loser_name']
#X = player_data[['loser_age']]
X = player_data.drop(columns=variables_to_drop)
Y = player_data[['match result']]

#change test size here
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=100)
random_forest.fit(X_train, Y_train.values.ravel())

y_predict_rf = random_forest.predict(X_test)
accuracy = accuracy_score(Y_test, y_predict_rf)*100
print('Accuracy is: ' + str(round(accuracy)) + '%')

# Predict probabilities REFER TO DECISION TREE FOR MORE DETAILED COMMENTS
y_proba_rf = random_forest.predict_proba(X_test)
win_probabilities = y_proba_rf[:, 1]
win_prob_series = pd.Series(win_probabilities, name='Win_Probability')

# Concatenate X_test, Y_test, and win_prob_series into a single DataFrame
end = pd.concat([X_test.reset_index(drop=True), Y_test.reset_index(drop=True), win_prob_series], axis=1)

# Print the resulting DataFrame
print(end)

## Visualize forest feature importances
#feature_importances = random_forest.feature_importances_
#sorted_indices = np.argsort(feature_importances)[::-1]
#plt.figure(figsize=(15, 10))
#plt.title('Feature Importances')
#plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
#plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
#plt.tight_layout()
#plt.show()#

