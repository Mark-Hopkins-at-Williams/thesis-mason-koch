TEAM_SIZE = 6
LOCKED_MOVES = ['outrage', 'petaldance', 'thrash']
STALL_MOVES = ['banefulbunker', 'detect', 'endure', 'kingsshield', 'protect', 'quickguard', 'spikyshield', 'wideguard']
FUTURE_MOVES = ['futuresight', 'doomdesire']
NOT_IMPLEMENTED_CONDITIONS = ['fly', 'flashfire', 'entrainment', 'skillswap']
# These are not implemented either.
CONSTANT_CONDITIONS = ['truant', 'arceus', 'silvally']
# The first list came from Pokemon-Showdown/data/statuses. The second list came from grep -r addVolatile in the Pokemon Showdown directory. As such, many of these statuses may not be relevant. But it should be a reasonably complete list.
POKEMON_SHOWDOWN_DATA_SATUSES = ['brn','par','slp','frz','psn','tox','confusion','flinch','trapped','partiallytrapped','lockedmove','twoturnmove',
'choicelock','mustrecharge','futuremove','healreplacement','stall','gem','raindance','primordialsea','sunnyday',
'desolateland','sandstorm','hail','deltastream','arceus','silvally']
GREP_ADDVOLATILE_STATUSES = ['disable','attract','flashfire','slowstart','truant','unburden','zenmode','focusenergy','metronome','micleberry',
'leppaberry','beakblast','gastroacid','counter','focuspunch','furycutter','iceball','lockon','magiccoat','mefirst',
'metalburst','mirrorcoat','perishsong','rollout','leechseed','shelltrap','throatchop', 'yawn']
# flinch is not a status which will be true when we are taking an action. Choicelock cannot actually be detected in battle. mustrecharge was not implemented because the moves which cause it suck. healreplacement is an obscure flag used for the z-move effects of memento and parting shot (source: Pokemon-Showdown/data/moves.js). i am not using z-moves so I did not implement this. gem is used for the effects of the gem items, so this is collinear with the item slot. Rain Dance through Delta Stream are weather. Arceus and Silvally are constant insofar as the neural network is concerned. disable was not implemented because it is irrelevant and it would be difficult to figure out which move it is targeting. flashfire was not implemented since there is no real way for Smogon to know if it is active. slow start was not implemented since regigigas is trash. Truant was not implemented because Slaking is trash. I was not sure how to implement unburden given the limitations of Smogon. Zen Mode is collinear with health and so was not implemented. i wanted very much to implement metronome because 6 Pokemon have a moveset recommending it, but I didn't see a way to do this quickly. leppaberry is collinear with item. beakblast, magic coat and mirror coat will not be true when we are taking an aciton. same with counter, focus punch, me first, metal burst and shell trap. Rollout was not implemented because it is bad.
# Not listed in the above: it was not clear how I would implement entrainment or skill swap. I consulted https://www.smogon.com/forums/threads/smogon-premier-league-xi-usage-statistics.3658913/ SS OU, and they were not used, so I did not implement them. Similarly, baton pass was not implemented.
# These status conditions are either on or off. You are either burned, or you are not.
ABSOLUTE_STATUS_CONDITIONS = ['brn', 'par', 'psn', 'trapped', 'lockedmove', 'twoturnmove'] + ['attract', 'focusenergy', 'micleberry', 'gastroacid', 'lockon', 'leechseed', 'substitute']
# These status conditions are time-based and go up over time.
RELATIVE_STATUS_CONDITIONS = ['slp', 'frz', 'tox', 'confusion', 'partiallytrapped', 'futuremove', 'stall'] + ['furycutter', 'iceball', 'throatchop', 'yawn', 'perish1', 'perish2', 'perish3']
REPEATED_STATUS_CONDITIONS = ['stall', 'furycutter', 'iceball', 'perish3', 'perish2', 'perish1']
WEATHER_STATUS_CONDITIONS = ['raindance', 'primordialsea', 'sunnyday', 'desolateland', 'sandstorm', 'hail', 'deltastream']
STATUS_LOOKUP = ABSOLUTE_STATUS_CONDITIONS + RELATIVE_STATUS_CONDITIONS
NUM_STATUS_CONDITIONS = len(STATUS_LOOKUP)
STATUS_DICT = {}
for i in range(NUM_STATUS_CONDITIONS):
    STATUS_DICT[STATUS_LOOKUP[i]] = i
for i in range(len(FUTURE_MOVES)):
    STATUS_DICT[FUTURE_MOVES[i]] = STATUS_DICT['futuremove']
for i in range(len(STALL_MOVES)):
    STATUS_DICT[STALL_MOVES[i]] = STATUS_DICT['stall']
for i in range(len(LOCKED_MOVES)):
    STATUS_DICT[LOCKED_MOVES[i]] = STATUS_DICT['lockedmove']

BOOST_DICT = {'atk': 0, 'def': 1, 'spa': 2, 'spd': 3, 'spe': 4, 'accuracy': 5, 'evasion': 6}
NUM_STAT_BOOSTS = len(BOOST_DICT) # three stages of attack boosts is represented as a 3
WEATHER_DICT = {'': 0, 'none':0, 'raindance': 1, 'primordialsea': 2, 'sunnyday': 3, 'desolateland': 4, 'sandstorm': 5, 'hail': 6, 'deltastream': 7}
WEATHER_LOOKUP = ['', 'raindance', 'primordialsea', 'sunnyday', 'desolateland', 'sandstorm', 'hail', 'deltastream']
NUM_WEATHER = len(WEATHER_LOOKUP)   # one-hot encoding
TERRAIN_DICT = {'': 0, 'electricterrain': 1, 'grassyterrain': 2, 'mistyterrain': 3, 'psychicterrain': 4}
# Sadly, terrain is denoted differently in the game code (above) and in the server messages (below).
TERRAIN_LOOKUP = {'move: Electric Terrain': 1, 'move: Grassy Terrain': 2, 'move: Misty Terrain': 3, 'move: Psychic Terrain': 4}
TERRLIST = ["no terrain", "electric terrain", "grassy terrain", "misty terrain", "psychic terrain"]
NUM_TERRAIN = len(TERRAIN_DICT)   # one-hot encoding
HAZARD_DICT = {'spikes': 0, 'toxicspikes': 1, 'stealthrock': 2, 'stickyweb': 3, 'auroraveil': 4, 'reflect': 5, 'lightscreen': 6}
HAZARD_DICT2 = {0: 'spikes', 1: 'toxicspikes', 2: 'stealthrock', 3: 'stickyweb', 4: 'auroraveil', 5: 'reflect', 6: 'lightscreen'}
NUM_HAZARDS = len(HAZARD_DICT)    # one-hot encoding
HAZARD_LOOKUP = {'move: spikes': 0, 'move: toxic spikes': 1, 'move: stealth rock': 2, 'move: sticky web': 3, 'move: aurora veil': 4, 'reflect': 5, 'move: light screen': 6}

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

OFFSET_HEALTH = 0    # Health is measured as a fraction of the total
OFFSET_STATUS_CONDITIONS = OFFSET_HEALTH + TEAM_SIZE * 2
OFFSET_STAT_BOOSTS = OFFSET_STATUS_CONDITIONS + TEAM_SIZE*NUM_STATUS_CONDITIONS*2
OFFSET_WEATHER = OFFSET_STAT_BOOSTS + NUM_STAT_BOOSTS*2
OFFSET_TERRAIN = OFFSET_WEATHER + NUM_WEATHER
OFFSET_HAZARDS = OFFSET_TERRAIN + NUM_TERRAIN
OFFSET_ITEM = OFFSET_HAZARDS + NUM_HAZARDS*2
OFFSET_TRICK_ROOM = OFFSET_ITEM + TEAM_SIZE * 2
OFFSET_GRAVITY = OFFSET_TRICK_ROOM + 1
N = OFFSET_GRAVITY + 1

