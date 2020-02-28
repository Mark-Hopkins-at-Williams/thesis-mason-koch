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
        # Add whether this Pokemon is active
        index = pokedex[name.lower()]['num']
        retval.append([index-1, pokemon['active']])
        if pokemon['active']:
            retval2 = ordinal_index
    # TODO: Refactor these for loops. They are basically identical.
    for i in range(6):
        # Append health information. This and the status information is almost identical to that for the Pokemon on our side.
        condition = mydict['State'][i].split(' ')
        if len(condition) == 1:
            # Our Pokemon is alive and has no status conditions
            health = int(condition[0].split('/')[0])/OUR_TEAM_MAXHEALTH[i]
            retval.append([OFFSET_HEALTH +  + i, health])
            for j in range(7): # TODO: REPLACE THIS 7???
                retval.append([OFFSET_STATUS_CONDITIONS +NUM_STATUS_CONDITIONS*(+i) + j, False])
        else:
            if condition[1] == 'fnt':
                # Our Pokemon has fainted. Serves them right!
                retval.append([OFFSET_HEALTH +  + i, 0])
                # Fainted pokemon have no status conditions
                for j in range(7): # TODO: REPLACE THIS 7???
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*(+i) + j, False])

            else:
                if len(condition) == 2:
                    condition.append(1)
                assert(len(condition) == 3)
                # Our Pokemon has a status condition
                assert(condition[1] in STATUS_DICT)
                for j in range(7): # TODO: REPLACE THIS 7???
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*( + i) + j, (STATUS_DICT[condition[1]] == j) *int(condition[2])])

                # Now add the health
                health = float(condition[0].split('/')[0])/OUR_TEAM_MAXHEALTH[i]
                retval.append([OFFSET_HEALTH +  + i, health])


    active_pokemon = int(mydict['State'][12])
    retval3 = active_pokemon

    for i in range(6):
        # Append health information. This and the status information is almost identical to that for the Pokemon on our side.
        condition = mydict['State'][i+TEAM_SIZE].split(' ')
        if len(condition) == 1:
            # Opposing Pokemon is alive and has no status conditions
            health = int(condition[0].split('/')[0])/OPPONENT_TEAM_MAXHEALTH[i]
            retval.append([OFFSET_HEALTH + TEAM_SIZE + i, health])
            for j in range(7): # TODO: REPLACE THIS 7???
                retval.append([OFFSET_STATUS_CONDITIONS +NUM_STATUS_CONDITIONS*(TEAM_SIZE+i) + j, False])
        else:
            if condition[1] == 'fnt':
                # Opposing Pokemon has fainted. Serves them right!
                retval.append([OFFSET_HEALTH + TEAM_SIZE + i, 0])
                # Fainted pokemon have no status conditions
                for j in range(7): # TODO: REPLACE THIS 7???
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*(TEAM_SIZE+i) + j, False])
            else:
                if len(condition) == 2:
                    condition.append(1)
                assert(len(condition) == 3)

                # Opposing Pokemon has a status condition
                assert(condition[1] in STATUS_DICT)
                for j in range(7): # TODO: REPLACE THIS 7???
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*(TEAM_SIZE + i) + j, (STATUS_DICT[condition[1]] == j) * int(condition[2])])
                # Now add the health
                health = float(condition[0].split('/')[0])/OPPONENT_TEAM_MAXHEALTH[i]
                retval.append([OFFSET_HEALTH + TEAM_SIZE + i, health])
        # Append which Pokemon is on the field
        index = pokedex[OPPONENT_TEAM[i]]['num']
        retval.append([NUM_POKEMON + index-1, i == active_pokemon])

    # Deal with 7 stat boosts on each team
    for i in [7,8,9,10,11,12,13, 14,15,16,17,18,19,20]:
        retval.append([OFFSET_STAT_BOOSTS + i-7, mydict['State'][i+6]])
    # Add weather
    for i in range(NUM_WEATHER):
        retval.append([OFFSET_WEATHER + i, i == WEATHER_DICT[mydict['State'][27]]])
    # And terrain
    for i in range(NUM_TERRAIN):
        retval.append([OFFSET_TERRAIN + i, i == TERRAIN_DICT[mydict['State'][28]]])
    # And entry hazards! Note that these are values like ['spikes', 'toxicspikes'].
    # Crucially, 'spikes' is not in ['toxicspikes'].
    for hazard, index in HAZARD_DICT.items():
        retval.append([OFFSET_HAZARDS+index, hazard in mydict['State'][29]])
        retval.append([OFFSET_HAZARDS+index+4, hazard in mydict['State'][30]])
    for i in range(TEAM_SIZE*2):
        retval.append([OFFSET_ITEM + i, mydict['State'][31+i] != ''])
    # We are returning a list of index-value pairs, where you look in to
    # the state and replace the thing at the index with the given value.
    return retval, retval2, retval3

