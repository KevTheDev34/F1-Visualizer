import fastf1

fastf1.set_log_level('DEBUG')
fastf1.Cache.enable_cache('cache')  # Enable caching to speed up future runs

session = fastf1.get_session(2024, 'Brazil', 'R')

session.load()  # Load the session data

print("Laps data size before modification: {}".format(session.laps.shape))  # Print the laps dataframe to see the raw data

# session.laps is a dataframe, can modify using pandas
laps = session.laps
laps = laps[laps['LapNumber'] != 1] # first lap not valid because of starting from grid times
print(len(laps))
laps = laps[laps['PitInTime'].isna()]
print(len(laps))
laps = laps[laps['PitOutTime'].isna()]
print(len(laps))
laps = laps[laps['TrackStatus'] == '1'] # Keep only fully green-flag laps (excludes any lap touched by yellow/SC/red/VSC)
print(len(laps))
laps = laps[laps['Deleted'] == False]
print(len(laps))
laps = laps[laps['LapTime'].notna()]
print(len(laps))

# for each driver, filter out the lap times that are over 107% of their median pace
driver_medians = laps.groupby('Driver')['LapTime'].transform('median')
clean_laps = laps[laps['LapTime'] <= driver_medians * 1.07]

print("Laps data size after modification: {}".format(clean_laps.shape))  # Print the modified laps dataframe to see the changes