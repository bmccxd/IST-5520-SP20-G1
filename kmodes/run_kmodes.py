import numpy as np
from kmodes.kmodes import KModes
import pandas as pd

# np.random.seed(50)
columns = ["Animal", "DayNight", "Impairment", "InjuryType", "RoadCharacteristics", "SurfaceCondition", "Weather"]
df = pd.read_csv("../combined.csv")[columns]#.sample(100)


print(df.shape)
for repl in ['nan', 'Unknown', 'Other - Explain in Narrative', 'Not Reported']:
  df = df.replace(repl, np.nan, regex=True)

df = df.dropna()
print(df.shape)

# for col in df:
#   print(df[col].unique())

km = KModes(n_clusters=4, init='Cao', verbose=1)

clusters = km.fit_predict(df)

# Print the cluster centroids
f = open("clusters.txt", 'w')
for col in df.columns:
  f.write('{0: <25}'.format(col))
f.write("\n")
for ii in km.cluster_centroids_:
  # these combinations of characteristics tend to be fairly common when accidents occur
  strin = ''
  for jj in ii:
    strin += '{0: <25}'.format(jj)
  print(strin)
  f.write(strin + "\n")
  
f.close()