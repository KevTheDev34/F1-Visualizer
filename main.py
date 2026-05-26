import fastf1

fastf1.Cache.enable_cache('cache')  # Enable caching to speed up future runs

session = fastf1.get_session(2026, 5, 5)  # Get the session for the 2026 Spanish Grand Prix

session.load()  # Load the session data

print(session)