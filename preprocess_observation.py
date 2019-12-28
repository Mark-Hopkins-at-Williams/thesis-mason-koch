""" preprocess_observation for pkmn.py. Kept in its own separate file in part because it is long. Functions like this generally exist for all games, but they differ a lot game to game. """
import json
from pokedex import pokedex
from game_model import *

def preprocess_observation(I):
    # There will always be a preprocessing step, but it will look different for different games.
    # In this case, we get back a stringified JSON object which ultimately hails from battle-stream in the Pokemon-Showdown/sim directory.
    # This gives us pretty much the whole state. In previous versions, this gave us the change in the state,
    # so preprocess_observation is still expected to return changes. Fixing this is on the todo list.
    # Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    mydict = json.loads(I)
    return preprocess_observation_helper(mydict)

def preprocess_observation_helper(mydict):
    retval = []
    retval2 = [-1,-1,-1,-1,-1,-1]
    k = 1
    for pokemon in mydict['side']['pokemon']:
        offset = NUM_POKEMON*2
        name = pokemon['ident'][4:]
        ordinal_index = OUR_TEAM[name.lower()]
        # Add health information for this pokemon
        condition = pokemon['condition'].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([offset + ordinal_index, health])
        offset += TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*ordinal_index
        # Add status information for this Pokemon
        if len(condition) != 1 and condition[1] != 'fnt':
            # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
            # TODO: confusion is not mutually exclusive with other status conditions, take account of this
            assert(len(condition) == 2)
            assert(condition[1] in STATUS_DICT)
            for i in range(7):
                retval.append([offset + i, STATUS_DICT[condition[1]] == i])
        else:
            for i in range(7):
                retval.append([offset + i, 0])
        # Add whether this Pokemon is active
        index = pokedex[name.lower()]['num']
        retval.append([index-1, pokemon['active']])
        retval2[ordinal_index] = k
        k += 1
    other_indices = [pokedex[OPPONENT_TEAM[0]]['num'], pokedex[OPPONENT_TEAM[1]]['num'], pokedex[OPPONENT_TEAM[2]]['num'], pokedex[OPPONENT_TEAM[3]]['num'], pokedex[OPPONENT_TEAM[4]]['num'], pokedex[OPPONENT_TEAM[5]]['num']]
    assert(int(mydict['State'][6]) != 6)  # Remove this eventually
    cur_pokemon = int(mydict['State'][6])
    for i in range(6):
        offset = NUM_POKEMON*2+TEAM_SIZE
        # Append health information
        condition = mydict['State'][i].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([offset + i, health])
        offset += NUM_STATUS_CONDITIONS*TEAM_SIZE + NUM_STATUS_CONDITIONS*i
        # Add status information
        if len(condition) != 1 and condition[1] != 'fnt':
            # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
            # TODO: confusion is not mutually exclusive with other status conditions, take account of this
            assert(len(condition) == 2)
            assert(condition[1] in STATUS_DICT)
            for j in range(7):
                retval.append([offset + j, STATUS_DICT[condition[1]] == j])
        else:
            for j in range(7):
                retval.append([offset + j, 0])
        # Append which Pokemon is on the field
        retval.append([NUM_POKEMON + other_indices[i]-1, i == cur_pokemon])
    # Deal with stat boosts
    offset = NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2
    for i in [7,8,9,10,11,12,13,   14,15,16,17,18,19,20]:
        retval.append([offset + i-7, mydict['State'][i]])
    # Add weather
    assert(mydict['State'][21] in WEATHER_DICT)
    offset += NUM_STAT_BOOSTS*2
    for i in range(6):
        retval.append([offset + i, i == WEATHER_DICT[mydict['State'][21]]])
    # And terrain
    assert(mydict['State'][22] in TERRAIN_DICT)
    offset += NUM_WEATHER
    for i in range(4):
        retval.append([offset + i, i == TERRAIN_DICT[mydict['State'][22]]])
    # And entry hazards!
    # Hope spikes and toxic spikes don't get confused
    offset += NUM_TERRAIN
    # mydict['State'][23] returns a value like ['spikes', 'toxicspikes'].
    # note that 'spikes' is not in ['toxicspikes'].
    retval.append([offset, 'spikes' in mydict['State'][23]])
    retval.append([offset+1, 'toxicspikes' in mydict['State'][23]])
    retval.append([offset+2, 'stealthrock' in mydict['State'][23]])
    retval.append([offset+3, 'stickyweb' in mydict['State'][23]])
    retval.append([offset+4, 'spikes' in mydict['State'][24]])
    retval.append([offset+5, 'toxicspikes' in mydict['State'][24]])
    retval.append([offset+6, 'stealthrock' in mydict['State'][24]])
    retval.append([offset+7, 'stickyweb' in mydict['State'][24]])
    #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    return retval, retval2

