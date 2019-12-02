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
    I = I.split('\n')
    mydict = json.loads(I[-3])
    return preprocess_observation_helper(mydict)

def preprocess_observation_helper(mydict):
    retval = []
    retval2 = [-1,-1,-1,-1,-1,-1]
    k = 1
    for pokemon in mydict['side']['pokemon']:
        name = pokemon['ident'][4:]
        tempDict = {'Aggron': 0, 'Arceus': 1, 'Cacturne': 2, 'Dragonite': 3, 'Druddigon': 4, 'Uxie': 5}
        # Add health information for this pokemon
        condition = pokemon['condition'].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([NUM_POKEMON*2 + tempDict[name], health])
        # Add status information for this Pokemon
        if len(condition) != 1 and condition[1] != 'fnt':
            # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
            # TODO: confusion is not mutually exclusive with other status conditions, take account of this
            assert(len(condition) == 2)
            assert(condition[1] in STATUS_DICT)
            for i in range(7):
                retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS * tempDict[name] + i, STATUS_DICT[condition[1]] == i])
        else:
            for j in range(7):
                retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS * tempDict[name] + j, 0])
        # Add whether this Pokemon is active
        index = pokedex[name.lower()]['num']
        retval.append([index-1, pokemon['active']])
        retval2[tempDict[name]] = k
        k += 1
    other_indices = [pokedex['houndoom']['num'], pokedex['arceus']['num'], pokedex['cacturne']['num'], pokedex['dragonite']['num'], pokedex['druddigon']['num'], pokedex['uxie']['num']]
    assert(int(mydict['State'][6]) != 6)  # Remove this eventually
    cur_pokemon = int(mydict['State'][6])
    for i in range(6):
        # Append health information
        condition = mydict['State'][i].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([NUM_POKEMON*2 + TEAM_SIZE + i, health])
        # Add status information
        if len(condition) != 1 and condition[1] != 'fnt':
            # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
            # TODO: confusion is not mutually exclusive with other status conditions, take account of this
            assert(len(condition) == 2)
            assert(condition[1] in STATUS_DICT)
            for j in range(7):
                retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE + NUM_STATUS_CONDITIONS * i + j, STATUS_DICT[condition[1]] == j])
        else:
            for j in range(7):
                retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE + NUM_STATUS_CONDITIONS * i + j, 0])

        # Append which Pokemon is on the field
        retval.append([NUM_POKEMON + other_indices[i]-1, i == cur_pokemon])
    # Deal with stat boosts
    for i in [7,8,9,10,11,12,13,   14,15,16,17,18,19,20]:
        retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + i-7, mydict['State'][i]])
    # Add weather
    assert(mydict['State'][21] in WEATHER_DICT)
    for i in range(6):
        retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + i, i == WEATHER_DICT[mydict['State'][21]]])
    # And terrain
    assert(mydict['State'][22] in TERRAIN_DICT)
    for i in range(4):
        retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + i, i == TERRAIN_DICT[mydict['State'][22]]])
    # And entry hazards!
    # Hope spikes and toxic spikes don't get confused
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN, 'spikes' in mydict['State'][23]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+1, 'toxicspikes' in mydict['State'][23]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+2, 'stealthrock' in mydict['State'][23]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+3, 'stickyweb' in mydict['State'][23]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+4, 'spikes' in mydict['State'][24]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+5, 'toxicspikes' in mydict['State'][24]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+6, 'stealthrock' in mydict['State'][24]])
    retval.append([NUM_POKEMON*2 + TEAM_SIZE*2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2 + NUM_WEATHER + NUM_TERRAIN+7, 'stickyweb' in mydict['State'][24]])
    #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    return retval, retval2

