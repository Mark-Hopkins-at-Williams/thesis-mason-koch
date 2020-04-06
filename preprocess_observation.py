""" preprocess_observation for env_pkmn. """
import json
from pokedex import pokedex
from game_model import *

def preprocess_observation(I):
    # The argument to this function is a stringified JSON object which ultimately hails from battle-stream in the Pokemon-Showdown/sim directory.
    mydict = json.loads(I)
    # We are returning a list of index-value pairs, where you look in to
    # the state and replace the thing at the index with the given value.
    retval = []
    RELEVANT_MAXHEALTH = OUR_TEAM_MAXHEALTH + OPPONENT_TEAM_MAXHEALTH
    # This for loop appends health and status information.
    for i in range(12):
        condition = mydict['State'][i].split(' ')
        if condition[0] == '0':
            # The Pokemon has fainted
            retval.append([OFFSET_HEALTH + i, 0])
            # Fainted pokemon have no status conditions
            for j in range(NUM_STATUS_CONDITIONS):
                retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, False])
        else:
            # The Pokemon hasn't fainted, so it must have health
            health = int(condition[0].split('/')[0])/RELEVANT_MAXHEALTH[i]
            assert(health > 0 and health <= 1.0)
            retval.append([OFFSET_HEALTH + i, health])
            if len(condition) == 1:
                # We have no status conditions
                for j in range(NUM_STATUS_CONDITIONS):
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, False])
            else:
                # We have at least one status condition
                for j in range(NUM_STATUS_CONDITIONS):
                    # For every status condition, say whether this Pokemon has that condition
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, STATUS_LOOKUP[j] in condition])
                for i in range(1, len(condition)):
                    # Make sure our status conditions are valid. This doesn't guarantee,
                    # for instance, that we don't have two of the same type of status (which would be a bug).
                    if condition[i] not in STATUS_DICT:
                        # STATUS_DICT contains all the status conditions we care about. FIrst, ask if it is a condition which is
                        # constant as far as our model is concerened.
                        if condition[i] not in CONSTANT_CONDITIONS:
                            # Allright, ask if it is a kind of weather.
                            if condition[i] not in WEATHER_STATUS_CONDITIONS:
                                if condition[i] not in NOT_IMPLEMENTED_CONDITIONS:
                                    # OK, something is definitely wrong.
                                    assert condition[i] in ALL_STATUS_CONDITIONS, "An invalid status condition was reported at position " + str(i) + " in condition " + str(condition)
                                    assert False, "A status condition was recognised, but reported as illegal at position " + str(i) + " in condition " + str(condition)

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
    retval.append([OFFSET_TRICK_ROOM, mydict['State'][43]])
    retval.append([OFFSET_GRAVITY, mydict['State'][44]])
    # We are also returning which Pokemon are active.
    for pokemon in mydict['side']['pokemon']:
        if pokemon['active']:
            retval2 = OUR_TEAM[pokemon['ident'][4:].lower()]
    retval3 = int(mydict['State'][12]) # Active Pokemon for the opposing team
    return retval, retval2, retval3

