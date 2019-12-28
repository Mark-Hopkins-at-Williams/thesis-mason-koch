""" preprocess_observation for pkmn.py. Kept in its own separate file in part because it is long. Functions like this generally exist for all games, but they differ a lot game to game. """
import numpy as np
from pokedex import pokedex
from game_model import *

# Todo in maybe December: Make this handle all the Pokemon and not just our favorites
p1a_indices = {}
p2a_indices = {}
combinedIndices = {}
for i in range(6):
    p1a_indices[OUR_TEAM[i]] = i
    p2a_indices[OPPONENT_TEAM[i]] = i
    combinedIndices[OUR_TEAM[i]] = NUM_POKEMON*2+i
    combinedIndices[OPPONENT_TEAM[i]] = NUM_POKEMON*2+i+6

def preprocess_observation(I):
    # There will always be a preprocessing step, but it will look different for different games.
    # In this case, the string we get back from the Pokemon simulator does not give us the entire state
    # of the game. Instead it gives us the change in the state. So return a list.
    # Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    retval = []
    I = I.split('\n')
    ci = 0
    retval2 = [-1,-1,-1,-1,-1,-1]
    for line in I:
        if ('switch|' in line) or ('drag|' in line):
            # There is a new Pokemon on the field. Update the pokemon on field and the health.
            if 'p1a' in line:
                temp = line.split('|')
                name = temp[2][5:].lower()
                index = pokedex[name]['num']
                # Not a very efficient solution
                for pokemon in p1a_indices:
                    newIndex = pokedex[pokemon.lower()]['num']
                    if newIndex != index:
                        retval.append([newIndex-1, 0])
                retval.append([index-1, 1])
                index = p1a_indices[name]
                condition = temp[4].split('/')[0].split(' ')
                health = int(condition[0])
                retval.append([OFFSET_HEALTH + index, health])
                if len(condition) != 1:
                    # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
                    # TODO: confusion is not mutually exclusive with other status conditions, take account of this
                    assert(len(condition) == 2)
                    assert(condition[1] in STATUS_DICT)
                    for i in range(7):
                        retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS * p1a_indices[name] + i, STATUS_DICT[condition[1]] == i])
                else:
                    for i in range(7):
                        retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS * p1a_indices[name] + i, 0])
            else:
                assert('p2a' in line)
                temp = line.split('|')
                name = temp[2][5:].lower()
                index = pokedex[name]['num']
                # Not a very efficient solution
                for pokemon in p2a_indices:
                    newIndex = pokedex[pokemon.lower()]['num']
                    if newIndex != index:
                        retval.append([newIndex-1+NUM_POKEMON, 0])
                retval.append([index-1+NUM_POKEMON, 1])
                index = p2a_indices[name]
                condition = temp[4].split('/')[0].split(' ')
                health = int(condition[0])
                retval.append([OFFSET_HEALTH + TEAM_SIZE + index, health])
                if len(condition) != 1:
                    # in the future, a numerical value (e.g. 2 turns of sleep remaining) would be nice instead of just 1/0.
                    # TODO: confusion is not mutually exclusive with other status conditions, take account of this
                    assert(len(condition) == 2)
                    assert(condition[1] in STATUS_DICT)
                    for i in range(7):
                        retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*TEAM_SIZE + NUM_STATUS_CONDITIONS * p2a_indices[name] + i, STATUS_DICT[condition[1]] == i])
                else:
                    for i in range(7):
                        retval.append([OFFSET_STATUS_CONDITIONS + NUM_STATUS_CONDITIONS*TEAM_SIZE + NUM_STATUS_CONDITIONS * p2a_indices[name] + i, 0])
        elif 'damage|' in line or 'heal|' in line:
            if 'Substitute' not in line:
                temp = line.split('|')
                name = temp[2][5:].lower()
                if temp[-1][0] == '[':
                    #The simulator is telling us the source of the damage
                    health = 0
                    if 'fnt' not in temp[-2]:
                        health = int(temp[-2].split('/')[0])
                    retval.append([combinedIndices[name], health])
                else:
                    if 'fnt' in temp[-1]:
                        health = 0
                        retval.append([combinedIndices[name], health])
                    else:
                        health = int(temp[-1].split('/')[0])
                        retval.append([combinedIndices[name], health])
        elif 'unboost|' in line:
            #Note: this gives relative boost, not absolute.
            temp = line.split('|')
            name = temp[2][5:].lower()
            retval.append(OFFSET_STAT_BOOSTS + ('p2a' in line)*NUM_STAT_BOOSTS + BOOST_DICT[temp[3]], -1 * float(temp[4]))
        elif 'boost|' in line:
            if 'Swarm' not in line:
                temp = line.split('|')
                name = temp[2][5:].lower()
                retval.append([OFFSET_STAT_BOOSTS + ('p2a' in line)*NUM_STAT_BOOSTS + BOOST_DICT[temp[3]], float(temp[4])])
        # TODO: make this less ugly?
        elif 'weather|' in line:
            # the weather has stopped
            if 'upkeep' in line:
                retval.append([OFFSET_WEATHER + 0, 1])
                retval.append([OFFSET_WEATHER + 1, 0])
                retval.append([OFFSET_WEATHER + 2, 0])
                retval.append([OFFSET_WEATHER + 3, 0])
                retval.append([OFFSET_WEATHER + 4, 0])
                retval.append([OFFSET_WEATHER + 5, 0])
                retval.append([OFFSET_WEATHER + 6, 0])
            else:
                temp = line.split('|')
                # The weather has started
                for i in range(7):
                    retval.append([OFFSET_WEATHER + i, i == WEATHER_DICT[temp[2][:-1].lower()]])
        elif 'fieldstart|' in line:
            if 'Electric' in line:
                retval.append([OFFSET_TERRAIN +0, 0])
                retval.append([OFFSET_TERRAIN +1, 1])
                retval.append([OFFSET_TERRAIN +2, 0])
                retval.append([OFFSET_TERRAIN +3, 0])
                retval.append([OFFSET_TERRAIN +4, 0])

            elif 'Grassy' in line:
                retval.append([OFFSET_TERRAIN +0, 0])
                retval.append([OFFSET_TERRAIN +1, 0])
                retval.append([OFFSET_TERRAIN +2, 1])
                retval.append([OFFSET_TERRAIN +3, 0])
                retval.append([OFFSET_TERRAIN +4, 0])

            elif 'Misty' in line:
                retval.append([OFFSET_TERRAIN +0, 0])
                retval.append([OFFSET_TERRAIN +1, 0])
                retval.append([OFFSET_TERRAIN +2, 0])
                retval.append([OFFSET_TERRAIN +3, 1])
                retval.append([OFFSET_TERRAIN +4, 0])

            else:
                assert('Psychic' in line)
                retval.append([OFFSET_TERRAIN +0, 0])
                retval.append([OFFSET_TERRAIN +1, 0])
                retval.append([OFFSET_TERRAIN +2, 0])
                retval.append([OFFSET_TERRAIN +3, 0])
                retval.append([OFFSET_TERRAIN +4, 1])
        elif 'sidestart|' in line:
            hazard = line.split('|')[-1][:-1].lower()
            retval.append([OFFSET_HAZARDS + HAZARD_DICT[hazard] + NUM_HAZARDS * ('p2' in line),1])
        elif 'sideend|' in line:
            hazard = line.split('|')[-1][:-1].lower()
            retval.append([OFFSET_HAZARDS + HAZARD_DICT[hazard] + NUM_HAZARDS * ('p2' in line),0])

        elif line == 'p1: Aggron\r':
            retval2[0] = ci
        elif line == 'p1: Arceus\r':
            retval2[1] = ci
        elif line == 'p1: Cacturne\r':
            retval2[2] = ci
        elif line == 'p1: Dragonite\r':
            retval2[3] = ci
        elif line == 'p1: Druddigon\r':
            retval2[4] = ci
        elif line == 'p1: Uxie\r':
            retval2[5] = ci
        ci += 1

    #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    print(retval)
    retval2 -= np.min(retval2)
    retval2 += 1
    return retval, retval2
