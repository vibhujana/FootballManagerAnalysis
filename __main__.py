from dis import dis
from turtle import pos
from webbrowser import get
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pip import main
from pyparsing import col
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import scipy.stats as stats
import csv
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


main_df = pd.read_csv("dataset.csv")
# fm17_df.reset_index(inplace=True)
main_df = main_df.set_index('UID')
attributes_df = main_df[["AerialAbility", "CommandOfArea", "Communication", "Eccentricity", "Handling", "Kicking", "OneOnOnes", "Reflexes", "RushingOut", "TendencyToPunch", "Throwing", "Corners", "Crossing", "Dribbling", "Finishing", "FirstTouch", "Freekicks", "Heading", "LongShots", "Longthrows", "Marking", "Passing", "PenaltyTaking",
                         "Tackling", "Technique", "Aggression", "Anticipation", "Bravery", "Composure", "Concentration", "Vision", "Decisions", "Determination", "Flair", "Leadership", "OffTheBall", "Positioning", "Teamwork", "Workrate", "Acceleration", "Agility", "Balance", "Jumping", "NaturalFitness", "Pace", "Stamina", "Strength", "LeftFoot", "RightFoot"]]
def get_wonderkids_unsupervised(df):
    df = df.drop(df[df["Age"] > 21].index)
    df = df[["Name","CurrentAbility","Consistency","ImportantMatches","InjuryProness","Versatility","Adaptability","Pressure","Professional","Determination"]]
    print(df.size)
    X = df.drop(["Name"]).values
    weights = np.ones(len(X))
    weights[:670] = 9
    kmeans = KMeans(n_clusters=2).fit(X,sample_weight=  weights)
    labels = kmeans.labels_
    plt.scatter(X[:, 1], X[:, 2], c=labels)
    plt.show()
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    print(dict(zip(labels, counts)))
    df["Label"] = kmeans.labels_
    return df
def attribute_to_ability(player):
    conversion = {1:1,2:1,3:1,4:1,5:1,6:1,7:20,8:40,9:60,10:80,11:100,12:120,13:140,14:160,15:180,16:199,17:200,18:200,19:200,20:200}
    to_convert = ['Corners','Crossing','Dribbling','Finishing','FirstTouch','Freekicks','Heading','LongShots','Longthrows','Marking','Passing','PenaltyTaking','Tackling','Technique','Aggression','Anticipation','Bravery','Composure','Concentration','Decisions','Determination','Flair','Leadership','OffTheBall','Positioning','Teamwork','Vision','Workrate','Acceleration','Agility','Balance','Jumping','NaturalFitness','Pace','Stamina','Strength','WeakerFoot','AerialAbility','CommandOfArea','Communication','Eccentricity','Handling','Kicking','OneOnOnes','TendencyToPunch','Reflexes','RushingOut','Throwing']
    for attribute in to_convert:
        player.at[player.index[0], attribute] = conversion[player.at[player.index[0], attribute]]
    return player
def calc_current_ability(player):
    player = pd.DataFrame([player.tolist()], columns=player.index)
    weights = (pd.read_csv("weights.csv"))  # .T
    weights = weights.set_index('Position')
    CA = 0
    player["WeakerFoot"] = player[["LeftFoot", "RightFoot"]].min(axis=1)
    player["GK"] = player["Goalkeeper"]
    player["SC"] = player["Striker"]
    player["MC"] = player["MidfielderCentral"]
    player["DM"] = player["DefensiveMidfielder"]
    player["AMC"] = player["AttackingMidCentral"]
    player["DC"] = player[["Sweeper", "DefenderCentral"]].max(axis=1)
    player["DRL"] = player[["DefenderLeft", "DefenderRight"]].max(axis=1)
    player["MRL"] = player[["MidfielderLeft", "MidfielderRight"]].max(axis=1)
    player["WBRL"] = player[["WingBackLeft", "WingBackRight"]].max(axis=1)
    player["AMRL"] = player[["AttackingMidLeft", "AttackingMidRight"]].max(axis=1)
    player = attribute_to_ability(player)
    avg_pos_CA = {'GK':0,'DRL':0,'DC':0,'WBRL':0,'DM':0,'MRL':0,'MC':0,'AMRL':0,'AMC':0,'SC':0}
    for position in (avg_pos_CA.keys()):
        temp = 0 
        for attribute in (player.columns):
            if (attribute not in weights.columns):
                continue
            attr = player.at[player.index[0], attribute] *  weights.at[position, attribute]
            temp += attr
        avg_pos_CA[position] = (temp/weights.loc[position].sum()) * 1.19
    CA = 0
    attr_weights = 0
    for position in avg_pos_CA.keys():
        if(player.at[player.index[0], position] > 17):
            CA += avg_pos_CA[position] *  (player.at[player.index[0], position])
            attr_weights += player.at[player.index[0], position]
    CA /= (attr_weights)
    return int(CA)

# main_df['CurrentAbility'] = main_df.apply(lambda row : calc_current_ability(row), axis = 1)
# print(main_df.head(100)["CurrentAbility"])
# main_df.to_csv("player_data.csv")
def extract_positions(str):
    positions = []
    position_groups = str.replace(" ","").split(",")
    for position_group in position_groups:
        current_positions = position_group[0:position_group.find("(")]
        if(current_positions == position_group):
            positions.append(current_positions)
        else:
            current_positions = current_positions.split("/")
            for position in current_positions:
                flanks = list(position_group[position_group.find("(")+1:position_group.find(")")])
                for flank in flanks:
                    positions.append(position+flank)
    return positions

def rtf_to_csv(input,output):
    with open(input, 'r') as rtf_file:
        csv_rows = []
        for line in rtf_file:
            if(bool(re.search(r'\d', line)) and bool(re.search(r'[a-zA-Z]', line))):
                csv_line = line.split('|')
                if(csv_line != "\"\n"):
                    csv_line = [cell.strip() for cell in csv_line]
                    csv_rows.append(csv_line)
    with open(output, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_rows)
def clean_2021_data(filename):
    Roles = {'Goalkeeper':"GG",'Striker':"STC",'AttackingMidCentral':"AMC",'AttackingMidLeft':"AML",'AttackingMidRight':"AMR",'DefenderCentral':"DC",'DefenderLeft':"DL",'DefenderRight':"DR",'DefensiveMidfielder':"DD",'MidfielderCentral':"MC",'MidfielderLeft':'ML','MidfielderRight':'MR','WingBackLeft':'WBL','WingBackRight':'WBR'}
    df = pd.read_csv(filename)
    df.drop(["Unnamed: 0","Inf","Name","Height","Caps","Goals","Yth Gls", "Rec", "Unnamed: 75","Yth Apps"],axis = 1,inplace=True)
    df['Positions'] = df.apply(lambda row : extract_positions(row["Position"]), axis = 1)
    for role in Roles.keys():
        df[role] = df["Positions"].apply(lambda x: 20 if Roles[role] in x else 1)
    df = df.replace(['Very Weak','Weak','Reasonable','Fairly Strong','Strong','Very Strong'],[1,6,10,14,16,20])
    df = df.set_index('UID')
    return df
df_2021 = clean_2021_data("player_data_2021.csv")
df_2021.to_csv("player_data_2021_updated.csv")



# df_mean = np.mean(main_df["CurrentAbility"])
# df_std = np.std(main_df["CurrentAbility"])
 
# # Calculating probability density function (PDF)
# pdf = stats.norm.pdf(main_df["CurrentAbility"].sort_values(), df_mean, df_std)

# # Drawing a graph
# plt.plot(main_df["CurrentAbility"].sort_values(), pdf)
# plt.xlim([30,70])  
# plt.xlabel("Current Ability", size=36)    
# plt.ylabel("Frequency", size=12)                
# plt.grid(True, alpha=0.3, linestyle="--")
# plt.show()
# print(main_df["CurrentAbility"].sort_values())

# def graph_stats_vs_age(df):
#     y_vars_list=['IntCaps','IntGoals','U21Caps','U21Goals','Height','Weight','AerialAbility','CommandOfArea','Communication','Eccentricity','Handling','Kicking','OneOnOnes','Reflexes','RushingOut','TendencyToPunch','Throwing','Corners','Crossing','Dribbling','Finishing','FirstTouch','Freekicks','Heading','LongShots','Longthrows','Marking','Passing','PenaltyTaking','Tackling','Technique','Aggression','Anticipation','Bravery','Composure','Concentration','Vision','Decisions','Determination','Flair','Leadership','OffTheBall','Positioning','Teamwork','Workrate','Acceleration','Agility','Balance','Jumping','LeftFoot','NaturalFitness','Pace','RightFoot','Stamina','Strength','Consistency','Dirtiness','ImportantMatches','InjuryProness','Versatility','Adaptability','Ambition','Loyalty','Pressure','Professional','Sportsmanship','Temperament','Controversy']
#     for i in range(0,len(y_vars_list)-3,4):
#         j = sns.pairplot(data=df.query('IntCaps > 1 & Age < 40'),
#                     y_vars=y_vars_list[i:i+4],
#                     x_vars=['Age'],
#                     kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
#         plt.show()
# df_without_goalkeeper  = main_df[main_df['PositionsDesc'] != 'GK ']
# graph_stats_vs_age(df_without_goalkeeper.drop(['Name','NationID','Born','PositionsDesc','Goalkeeper','Sweeper','Striker','AttackingMidCentral','AttackingMidLeft','AttackingMidRight','DefenderCentral','DefenderLeft','DefenderRight','DefensiveMidfielder','MidfielderCentral','MidfielderLeft','MidfielderRight','WingBackLeft','WingBackRight'],axis=1))


# def test():
#     predicted = kmeans.predict(attributes_df.drop("Segment K-means",axis=1))
#     dist_centers = kmeans.transform(attributes_df.drop("Segment K-means",axis=1))
#     print(np.argsort(dist_centers,axis=1)[0][0])


# test()
