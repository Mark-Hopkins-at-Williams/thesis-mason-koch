TEAM_SIZE = 6               # health, measured as a number
STATUS_DICT = {'brn': 0, 'par': 1, 'slp': 2, 'frz': 3, 'psn': 4, 'tox': 5, 'tox\r': 5, 'confusion': 6} 
STATUS_LOOKUP = ['brn', 'par', 'slp', 'frz', 'psn', 'tox', 'confusion']
NSC_PLACEHOLDER = 7    # status conditions are a one-hot encoding.
NUM_STATUS_CONDITIONS = 28  # from stream.battle.dataCache.Statuses in battle_stream.
                            # many of them unused because they don't look like status conditions.
                            # also, some of them were introduced in games I haven't played and
                            # am not familiar with. So, this number is not final.
BOOST_DICT = {'atk': 0, 'def': 1, 'spa': 2, 'spd': 3, 'spe': 4, 'accuracy': 5, 'evasion': 6}
NUM_STAT_BOOSTS = len(BOOST_DICT) # three stages of attack boosts is represented as a 3
WEATHER_DICT = {'': 0, 'raindance': 1, 'primordialsea': 2, 'sunnyday': 3, 'desolateland': 4, 'sandstorm': 5, 'hail': 6, 'deltastream': 7}
WEATHER_LOOKUP = ['', 'raindance', 'primordialsea', 'sunnyday', 'desolateland', 'sandstorm', 'hail', 'deltastream']
NUM_WEATHER = len(WEATHER_DICT)   # one-hot encoding
TERRAIN_DICT = {'': 0, 'electricterrain': 1, 'grassyterrain': 2, 'mistyterrain': 3, 'psychicterrain': 4}
# Sadly, terrain is denoted differently in the game code (above) and in the server messages (below).
TERRAIN_LOOKUP = {'move: Electric Terrain': 1, 'move: Grassy Terrain': 2, 'move: Misty Terrain': 3, 'move: Psychic Terrain': 4}
TERRLIST = ["no terrain", "electric terrain", "grassy terrain", "misty terrain", "psychic terrain"]
NUM_TERRAIN = len(TERRAIN_DICT)   # one-hot encoding
HAZARD_DICT = {'spikes': 0, 'toxicspikes': 1, 'stealthrock': 2, 'stickyweb': 3}
HAZARD_DICT2 = {0: 'spikes', 1: 'toxicspikes', 2: 'stealthrock', 3: 'stickyweb'}
NUM_HAZARDS = len(HAZARD_DICT)    # one-hot encoding
HAZARD_LOOKUP = {'move: spikes': 0, 'move: toxic spikes': 1, 'move: stealth rock': 2, 'move: sticky web': 3}

# Right now, the teams in test_random_player are not alphabetised. This is an issue because the following code
# assumes they are. Furthermore, they are not going to be alphabetised before 6v6 because the current Pokemon
# we are training on are not the alphabetically first ones. But, when 6v6 happens, the following code should
# go in to test_random_player and extract the teams we are playing for us.
"""
f = open('/Pokemon-Showdown/sim/examples/test_random_player.ts', 'r')
line = ''
while 'team: ' not in line:
    line = f.readline()
line = line.split(']')
line[0] = line[0][8:]    # Cut out '\tteam: '
for l in line:
    l = l.split('||')[0].lower()
OUR_TEAM = {line[0]:0, line[1]:1, line[2]:2, line[3]:3, line[4]:4, line[5]:5}
while 'team: ' not in line:
    line = f.readline()
line = line.split(']')
line[0] = line[0][8:]    # Cut out '\tteam: '
for l in line:
    l = l.split('||')[0].lower()
OPPONENT_TEAM = {line[0]:0, line[1]:1, line[2]:2, line[3]:3, line[4]:4, line[5]:5}
f.close()
"""

# Generally, if these teams to not match the teams provided in Pokemon-Showdown/sim/examples/test_random_player,
# Pokemon Showdown will crash due to a key error.
#OUR_TEAM = {'aggron':0, 'arceus':1, 'cacturne':2, 'dragonite':3, 'druddigon':4, 'uxie':5, 0:'aggron', 1:'arceus', 2:'cacturne', 3:'dragonite', 4:'druddigon', 5:'uxie'}
OUR_TEAM = {'houndoom':0, 'ledian':1, 'lugia':2, 'malamar':3, 'swellow':4, 'victreebel':5, 0:'houndoom', 1:'ledian', 2:'lugia', 3:'malamar', 4:'swellow', 5:'victreebel'}
#OPPONENT_TEAM = {'houndoom':0, 'ledian':1, 'lugia':2, 'malamar':3, 'swellow':4, 'victreebel':5, 0:'houndoom', 1:'ledian', 2:'lugia', 3:'malamar', 4:'swellow', 5:'victreebel'}
OPPONENT_TEAM = {'aggron':0, 'arceus':1, 'cacturne':2, 'dragonite':3, 'druddigon':4, 'uxie':5, 0:'aggron', 1:'arceus', 2:'cacturne', 3:'dragonite', 4:'druddigon', 5:'uxie'}
# Irrelevant for training, but relevant for Smogon.
OUR_TEAM_MAXHEALTH = [312, 272, 0, 334, 282, 0]
#OUR_TEAM_MAXHEALTH = [302, 402, 0, 343, 0, 312]
OPPONENT_TEAM_MAXHEALTH = [302, 402, 0, 343, 0, 312]
#OPPONENT_TEAM_MAXHEALTH = [312, 272, 0, 334, 282, 0]

POSSIBLE_ACTIONS = ["move 1", "move 2", "move 3", "move 4", "switch " + OUR_TEAM[0], "switch " + OUR_TEAM[1], "switch " + OUR_TEAM[2], "switch " + OUR_TEAM[3], "switch " + OUR_TEAM[4], "switch " + OUR_TEAM[5]]
OPPONENT_POSSIBLE_ACTIONS = ["move 1", "move 2", "move 3", "move 4", "switch " + OPPONENT_TEAM[0], "switch " + OPPONENT_TEAM[1], "switch " + OPPONENT_TEAM[2], "switch " + OPPONENT_TEAM[3], "switch " + OPPONENT_TEAM[4], "switch " + OPPONENT_TEAM[5]]
NUM_MOVES = 4


OFFSET_HEALTH = 0
OFFSET_STATUS_CONDITIONS = OFFSET_HEALTH + TEAM_SIZE * 2
OFFSET_STAT_BOOSTS = OFFSET_STATUS_CONDITIONS + TEAM_SIZE*NUM_STATUS_CONDITIONS*2
OFFSET_WEATHER = OFFSET_STAT_BOOSTS + NUM_STAT_BOOSTS*2
OFFSET_TERRAIN = OFFSET_WEATHER + NUM_WEATHER
OFFSET_HAZARDS = OFFSET_TERRAIN + NUM_TERRAIN
OFFSET_ITEM = OFFSET_HAZARDS + NUM_HAZARDS*2
N = OFFSET_ITEM + TEAM_SIZE*2


