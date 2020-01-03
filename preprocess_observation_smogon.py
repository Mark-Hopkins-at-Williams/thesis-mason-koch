""" preprocess_observation for env_pkmn_smogon. Kept in its own separate file because it is long and has very little repeated code. """
import numpy as np
from pokedex import pokedex
from game_model import *

p1a_indices = {}
p2a_indices = {}
combinedIndices = {}
for i in range(6):
    p1a_indices[OUR_TEAM[i]] = i
    p1a_indices[i] = pokedex[OUR_TEAM[i]]['num']
    p2a_indices[OPPONENT_TEAM[i]] = i
    p2a_indices[i] = pokedex[OPPONENT_TEAM[i]]['num']
    combinedIndices[OUR_TEAM[i]] = NUM_POKEMON*2+i
    combinedIndices[OPPONENT_TEAM[i]] = NUM_POKEMON*2+i+6

def preprocess_observation(I):
    # In this case, the string we get back from the Pokemon simulator does not give us the entire state
    # of the game. Instead it gives us the change in the state. So return a list.
    # Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    retval = []
    I = I.split('\n')
    ci = 0
    for line in I:
        split_line = line.split('|')
        if ('switch|' in line) or ('drag|' in line):
            # There is a new Pokemon on the field.
            if 'p1a' in line:
                # This relevant_indices solution is not markedly more elegant than what it replaced.
                # In that respect, despite the rewrite, this section is still unsatisfying.
                # It should be easier to maintain, however.
                relevant_indices = p1a_indices
                relevant_offsets = [0,0,0]
            else:
                assert('p2a' in line)
                relevant_indices = p2a_indices
                relevant_offsets = [NUM_POKEMON, TEAM_SIZE, NUM_STATUS_CONDITIONS*TEAM_SIZE]
            # Update the pokemon on field
            name = split_line[2][5:].lower()
            index = pokedex[name]['num']
            for i in range(6):
                retval.append([relevant_indices[i]-1 + relevant_offsets[0], relevant_indices[i] == index])
            # And the health
            condition = split_line[4].split('/')[0].split(' ')
            health = int(condition[0])
            retval.append([OFFSET_HEALTH + relevant_offsets[1] + relevant_indices[name], health])
            # And the status conditions (or lack thereof)
            if len(condition) != 1:
                assert(len(condition) == 2)
                for i in range(NSC_PLACEHOLDER):
                    retval.append([OFFSET_STATUS_CONDITIONS + relevant_offsets[2] + NUM_STATUS_CONDITIONS * relevant_indices[name] + i, STATUS_DICT[condition[1]] == i])
            else:
                for i in range(NSC_PLACEHOLDER):
                    retval.append([OFFSET_STATUS_CONDITIONS + relevant_offsets[2] + NUM_STATUS_CONDITIONS * relevant_indices[name] + i, 0])
        elif 'damage|' in line or 'heal|' in line:
            if 'Substitute' not in line:
                name = split_line[2][5:].lower()
                if split_line[-1][0] == '[':
                    # The simulator is telling us the source of the damage
                    health = 0
                    if 'fnt' not in split_line[-2]:
                        health = int(split_line[-2].split('/')[0])
                    retval.append([combinedIndices[name], health])
                else:
                    if 'fnt' in split_line[-1]:
                        health = 0
                        retval.append([combinedIndices[name], health])
                    else:
                        health = int(split_line[-1].split('/')[0])
                        retval.append([combinedIndices[name], health])
        elif 'unboost|' in line:
            # Note: this gives relative boost, not absolute.
            name = split_line[2][5:].lower()
            retval.append(OFFSET_STAT_BOOSTS + ('p2a' in line)*NUM_STAT_BOOSTS + BOOST_DICT[split_line[3]], -1 * float(temp[4]))
        elif 'boost|' in line:
            if 'Swarm' not in line:
                name = split_line[2][5:].lower()
                retval.append([OFFSET_STAT_BOOSTS + ('p2a' in line)*NUM_STAT_BOOSTS + BOOST_DICT[split_line[3]], float(temp[4])])
        elif 'weather|' in line:
            if 'upkeep' in line:
                # the weather has stopped
                retval.append([OFFSET_WEATHER, 1])
                for i in range(1, NUM_WEATHER):
                    retval.append([OFFSET_WEATHER + i, 0])
            else:
                # The weather has started
                retval.append([OFFSET_WEATHER, 0])
                for i in range(1, NUM_WEATHER):
                    retval.append([OFFSET_WEATHER + i, i == WEATHER_DICT[split_line[2][:-1].lower()]])
        elif 'fieldstart|' in line:
            retval.append([OFFSET_TERRAIN +0, 0])
            for i in range(1, NUM_TERRAIN):
                retval.append([OFFSET_TERRAIN+i, TERRAIN_LOOKUP[split_line[2]] == i])
        elif 'fieldend|' in line:
            retval.append([OFFSET_TERRAIN, 1])
            for i in range(1,NUM_TERRAIN):
                retval.append([OFFSET_TERRAIN+i, 0])
        elif 'sidestart|' in line:
            hazard = split_line[-1][:-1].lower()
            retval.append([OFFSET_HAZARDS + HAZARD_DICT[hazard] + NUM_HAZARDS * ('p2' in line), 1])
        elif 'sideend|' in line:
            hazard = split_line[-1][:-1].lower()
            retval.append([OFFSET_HAZARDS + HAZARD_DICT[hazard] + NUM_HAZARDS * ('p2' in line), 0])

    print(retval)
    return retval
