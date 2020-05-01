from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import numpy as np

columns = ["Animal", "DayNight", "Impairment", "InjuryType", "RoadCharacteristics", "SurfaceCondition", "Weather"]
df = pd.read_csv("../combined.csv")[columns]

for repl in ['nan', 'Unknown', 'Other - Explain in Narrative', 'Not Reported']:
  df = df.replace(repl, np.nan, regex=True)

df = df.dropna()

df2 = pd.get_dummies(data=df, columns=columns)
print(df2)

frequent_itemsets = apriori(df2, min_support=0.001, use_colnames=True)
print(frequent_itemsets.head())
print(frequent_itemsets.info())

frequent_itemsets.describe()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("Writing to csv now...")
rules.to_csv("all_rules.csv")

bestrules = rules[(rules['lift'] >= 9) & (rules['confidence'] >= 0.7)].sort_values(by='lift', ascending=False)
bestrules.to_csv("best_rules.csv")

# interesting rules I found (in all_rules.csv):
# row 249:  surface type = slush => property damage only with 0.809554551 confidence
# row 0:    animal type = deer => night with 0.559960357 confidence
# row 6:    animal type = deer => not at a junction with 0.955401388 confidence
# row 126:  impairment = alchold => night with 0.665167416 confidence
# row 155:  injury type = fatal => impairment = alcohol with 0.34741784 confidence
# row 208 : injury type = fatal => surface condition = dry with 0.765258216 confidence