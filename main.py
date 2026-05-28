import fastf1
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

fastf1.set_log_level('DEBUG')
fastf1.Cache.enable_cache('cache')  # Enable caching to speed up future runs

session = fastf1.get_session(2025, 'Hungary', 'R')

session.load()  # Load the session data

# print("Laps data size before modification: {}".format(session.laps.shape))  # Print the laps dataframe to see the raw data

# session.laps is a dataframe, can modify using pandas
laps = session.laps
laps = laps[laps['LapNumber'] != 1] # first lap not valid because of starting from grid times
# print(len(laps))
laps = laps[laps['PitInTime'].isna()]
# print(len(laps))
laps = laps[laps['PitOutTime'].isna()]
# print(len(laps))
laps = laps[laps['TrackStatus'] == '1'] # Keep only fully green-flag laps (excludes any lap touched by yellow/SC/red/VSC)
# print(len(laps))
laps = laps[laps['Deleted'] == False]
# print(len(laps))
laps = laps[laps['LapTime'].notna()]
# print(len(laps))

# for each driver, filter out the lap times that are over 107% of their median pace
driver_medians = laps.groupby('Driver')['LapTime'].transform('median')
clean_laps = laps[laps['LapTime'] <= driver_medians * 1.07].copy()

# print("Laps data size after modification: {}".format(clean_laps.shape))  # Print the modified laps dataframe to see the changes
clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()  # Convert LapTime to seconds for easier calculations
features = clean_laps.groupby('Driver').agg(
    median_pace = ('LapTimeSeconds', 'median'),
    pace_std = ('LapTimeSeconds', 'std'),
    best_lap_delta=('LapTimeSeconds', lambda x: x.min() - x.median()))

# print(features)
# print(features.isna().sum())  # Check for any NaN values in the features dataframe

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# print(scaled_features)

linked = linkage(scaled_features, method='ward')

dendogram = dendrogram(linked, labels=features.index.tolist())
# plt.title('Hierarchical Clustering Dendrogram')
# plt.tight_layout()
# plt.show()

clustering = AgglomerativeClustering(n_clusters=3)
cluster_labels = clustering.fit_predict(scaled_features)
features['Cluster'] = cluster_labels
print(features[['Cluster']])
