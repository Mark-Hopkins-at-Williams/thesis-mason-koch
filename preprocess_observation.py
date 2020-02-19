""" preprocess_observation for env_pkmn. """
import json
from pokedex import pokedex
from game_model import *

def preprocess_observation(I):
    # The argument to this function is a stringified JSON object which ultimately hails from battle-stream in the Pokemon-Showdown/sim directory.
    mydict = json.loads(I)
    retval = []
    retval2 = ''
    retval3 = ''
    for pokemon in mydict['side']['pokemon']:
        name = pokemon['ident'][4:]
        ordinal_index = OUR_TEAM[name.lower()]
        # Add health information for this pokemon
        # THIS CAN BE REFACTORED
        condition = pokemon['condition'].split(' ')
        if len(condition) == 1:
            # Our Pokemon is alive and has no status conditions
            health = int(condition[0].split('/')[0])
            retval.append([OFFSET_HEALTH + ordinal_index, health])
        else:
            assert(len(condition) == 2)
            if condition[1] == 'fnt':
                # Pokemon has fainted. Sad!
                retval.append([OFFSET_HEALTH + ordinal_index, 0])
            else:
                # Our Pokemon has a status condition
                assert(condition[1] in STATUS_DICT)
                for i in range(7):
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*ordinal_index + i, STATUS_DICT[condition[1]] == i])
                # Now add the health
                health = int(condition[0].split('/')[0])
                retval.append([OFFSET_HEALTH + ordinal_index, health])
        # Add whether this Pokemon is active
        index = pokedex[name.lower()]['num']
        retval.append([index-1, pokemon['active']])
        if pokemon['active']:
            retval2 = ordinal_index

    active_pokemon = int(mydict['State'][6])
    retval3 = active_pokemon

    for i in range(6):
        # Append health information. This and the status information is almost identical to that for the Pokemon on our side.
        condition = mydict['State'][i].split(' ')
        if len(condition) == 1:
            # Opposing Pokemon is alive and has no status conditions
            health = int(condition[0].split('/')[0])
            retval.append([OFFSET_HEALTH + TEAM_SIZE + i, health])
        else:
            assert(len(condition) == 2)
            if condition[1] == 'fnt':
                # Opposing Pokemon has fainted. Serves them right!
                retval.append([OFFSET_HEALTH + TEAM_SIZE + i, 0])
            else:
                # Opposing Pokemon has a status condition
                assert(condition[1] in STATUS_DICT)
                for j in range(7): # TODO: REPLACE THIS 7???
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, STATUS_DICT[condition[1]] == j])
                # Now add the health
                health = int(condition[0].split('/')[0])
                retval.append([OFFSET_HEALTH + TEAM_SIZE + i, health])
        # Append which Pokemon is on the field
        index = pokedex[OPPONENT_TEAM[i]]['num']
        retval.append([NUM_POKEMON + index-1, i == active_pokemon])

    # Deal with 7 stat boosts on each team
    for i in [7,8,9,10,11,12,13, 14,15,16,17,18,19,20]:
        retval.append([OFFSET_STAT_BOOSTS + i-7, mydict['State'][i]])
    # Add weather
    for i in range(6):
        retval.append([OFFSET_WEATHER + i, i == WEATHER_DICT[mydict['State'][21]]])
    # And terrain
    for i in range(4):
        retval.append([OFFSET_TERRAIN + i, i == TERRAIN_DICT[mydict['State'][22]]])
    # And entry hazards! Note that these are values like ['spikes', 'toxicspikes'].
    # Crucially, 'spikes' is not in ['toxicspikes'].
    for hazard, index in HAZARD_DICT.items():
        retval.append([OFFSET_HAZARDS+index, hazard in mydict['State'][23]])
        retval.append([OFFSET_HAZARDS+index+4, hazard in mydict['State'][24]])
    # We are returning a list of index-value pairs, where you look in to
    # the state and replace the thing at the index with the given value.
    return retval, retval2, retval3

