TEAM_SIZE = 6               # health, measured as a fraction of the total
# The first list came from Pokemon-Showdown/data/statuses. The second list came from grep -r addVolatile in the Pokemon Showdown directory. As such, many of these statuses may not be relevant. But it should be a reasonably complete list.
POKEMON_SHOWDOWN_DATA_SATUSES = ['brn','par','slp','frz','psn','tox','confusion','flinch','trapped','partiallytrapped','lockedmove','twoturnmove',
'choicelock','mustrecharge','futuremove','healreplacement','stall','gem','raindance','primordialsea','sunnyday',
'desolateland','sandstorm','hail','deltastream','arceus','silvally']
GREP_ADDVOLATILE_STATUSES = ['disable','attract','flashfire','slowstart','truant','unburden','zenmode','focusenergy','metronome','micleberry',
'leppaberry','beakblast','gastroacid','counter','focuspunch','furycutter','iceball','lockon','magiccoat','mefirst',
'metalburst','mirrorcoat','perishsong','rollout','leechseed','shelltrap','throatchop', 'yawn']
# flinch is not a status which will be true when we are taking an action. Not sure about lockedmove and twoturnmove, but I included them anyway. Choicelock might be irrelevant since it just disqualifies some moves, but I included it anyway. mustrecharge shouldn't be true when we are taking an action. healreplacement is an obscure flag used for the z-move effects of memento and parting shot (source: Pokemon-Showdown/data/moves.js). I am not using z-moves. gem is used for the effects of the gem items, so this is collinear with the item slot. Rain Dance through Delta Stream are weather. Arceus, Silvally and Truant are constant insofar as the neural network is concerned. Zen Mode is collinear with health but included anyway. leppaberry is collinear with item. beakblast, magic coat and mirror coat will not be true when we are taking an aciton. same with counter, focus punch, me first, metal burst, shell trap.
ALL_STATUS_CONDITIONS = POKEMON_SHOWDOWN_DATA_SATUSES + GREP_ADDVOLATILE_STATUSES
# These status conditions are either on or off. You are either burned, or you are not.
ABSOLUTE_STATUS_CONDITIONS = ['brn', 'par', 'psn', 'flinch', 'trapped', 'lockedmove', 'twoturnmove', 'choicelock'] + ['attract', 'flashfire', 'unburden', 'zenmode', 'focusenergy', 'micleberry', 'gastroacid', 'lockon', 'leechseed']
# These status conditions are time-based and go up over time.
RELATIVE_STATUS_CONDITIONS = ['slp', 'frz', 'tox', 'confusion', 'partiallytrapped', 'futuremove', 'stall'] + ['disable', 'slowstart', 'metronome', 'furycutter', 'iceball', 'perishsong', 'rollout', 'throatchop', 'yawn']
# These conditions are legal, but not included in our model.
CONSTANT_CONDITIONS = ['truant', 'arceus', 'silvally']
WEATHER_STATUS_CONDITIONS = ['raindance', 'primordialsea', 'sunnyday', 'desolateland', 'sandstorm', 'hail', 'deltastream']
STATUS_LOOKUP = ABSOLUTE_STATUS_CONDITIONS + RELATIVE_STATUS_CONDITIONS
NUM_STATUS_CONDITIONS = len(STATUS_LOOKUP)
STATUS_DICT = {}
for i in range(NUM_STATUS_CONDITIONS):
    STATUS_DICT[STATUS_LOOKUP[i]] = i

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

FNAME = "swellow_aggron"
# Generally, if these teams to not match the teams provided in Pokemon-Showdown/sim/examples/test_random_player,
# Pokemon Showdown will crash due to a key error.
if FNAME == "swellow_aggron":
    OUR_TEAM = {'houndoom':0, 'ledian':1, 'lugia':2, 'malamar':3, 'swellow':4, 'victreebel':5, 0:'houndoom', 1:'ledian', 2:'lugia', 3:'malamar', 4:'swellow', 5:'victreebel'}
    OPPONENT_TEAM = {'aggron':0, 'arceus':1, 'cacturne':2, 'dragonite':3, 'druddigon':4, 'uxie':5, 0:'aggron', 1:'arceus', 2:'cacturne', 3:'dragonite', 4:'druddigon', 5:'uxie'}
    # Irrelevant for training, but relevant for Smogon.
    OUR_TEAM_MAXHEALTH = [312, 272, 0, 334, 282, 0]
    OPPONENT_TEAM_MAXHEALTH = [302, 402, 0, 344, 0, 312]
elif FNAME == "aggron_swellow":
    OUR_TEAM = {'aggron':0, 'arceus':1, 'cacturne':2, 'dragonite':3, 'druddigon':4, 'uxie':5, 0:'aggron', 1:'arceus', 2:'cacturne', 3:'dragonite', 4:'druddigon', 5:'uxie'}
    OPPONENT_TEAM = {'houndoom':0, 'ledian':1, 'lugia':2, 'malamar':3, 'swellow':4, 'victreebel':5, 0:'houndoom', 1:'ledian', 2:'lugia', 3:'malamar', 4:'swellow', 5:'victreebel'}
    OUR_TEAM_MAXHEALTH = [302, 402, 0, 344, 0, 312]
    OPPONENT_TEAM_MAXHEALTH = [312, 272, 0, 334, 282, 0]

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
OFFSET_TRICK_ROOM = OFFSET_ITEM + TEAM_SIZE * 2
OFFSET_GRAVITY = OFFSET_TRICK_ROOM + 1
N = OFFSET_GRAVITY + 1

