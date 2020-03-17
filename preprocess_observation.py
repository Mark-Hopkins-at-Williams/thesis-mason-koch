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
        if len(condition) == 1:
            # The Pokemon is alive
            health = int(condition[0].split('/')[0])/RELEVANT_MAXHEALTH[i]
            retval.append([OFFSET_HEALTH + i, health])
            # And has no status conditions
            for j in range(NSC_PLACEHOLDER):
                retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, False])
        else:
            if condition[1] == 'fnt':
                # The Pokemon has fainted, so its health is zero
                retval.append([OFFSET_HEALTH + i, 0])
                # Fainted pokemon have no status conditions either
                for j in range(NSC_PLACEHOLDER):
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, False])
            else:
                # If the status condition doesn't have a timer, declare the timer to be 1
                if len(condition) == 2:
                    condition.append(1)
                assert(len(condition) == 3)
                assert(condition[1] in STATUS_DICT)
                for j in range(NSC_PLACEHOLDER):
                    retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*i + j, (STATUS_DICT[condition[1]] == j) * int(condition[2])])
                # Now add the health
                health = float(condition[0].split('/')[0])/RELEVANT_MAXHEALTH[i]
                retval.append([OFFSET_HEALTH + i, health])
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
    
    # We are also returning which Pokemon are active.
    for pokemon in mydict['side']['pokemon']:
        if pokemon['active']:
            retval2 = OUR_TEAM[pokemon['ident'][4:].lower()]
    retval3 = int(mydict['State'][12]) # Active Pokemon for the opposing team
    return retval, retval2, retval3

