NUM_POKEMON = 809           # covered in pokedex, which is too big to import here
TEAM_SIZE = 6
NUM_STATUS_CONDITIONS = 28  # from stream.battle.dataCache.Statuses in battle_stream.
                            # many of them unused because they don't look like status conditions.
                            # also, some of them were introduced in games I haven't played and
                            # am not familiar with. So, this number is not final.
STATUS_DICT = {'brn': 0, 'par': 1, 'slp': 2, 'frz': 3, 'psn': 4, 'tox': 5, 'confusion': 6}
NUM_STAT_BOOSTS = 7
BOOST_DICT = {'atk': 0, 'def': 1, 'spa': 2, 'spd': 3, 'spe': 4, 'accuracy': 5, 'evasion': 6}
NUM_WEATHER = 8
WEATHER_DICT = {'': 0, 'raindance': 1, 'primordialsea': 2, 'sunnyday': 3, 'desolateland': 4, 'sandstorm': 5, 'hail': 6, 'deltastream': 7}
NUM_TERRAIN = 5
TERRAIN_DICT = {'': 0, 'electricterrain': 1, 'grassyterrain': 2, 'mistyterrain': 3, 'psychicterrain': 4}
n = NUM_POKEMON * 2 + TEAM_SIZE*2 + TEAM_SIZE*NUM_STATUS_CONDITIONS*28 + NUM_STAT_BOOSTS * 2 + NUM_WEATHER + NUM_TERRAIN


