""" preprocess_observation for pkmn.py. Kept in its own separate file in part because it is long. Functions like this generally exist for all games, but they differ a lot game to game. """
import numpy as np
from pokedex import pokedex


# Todo in maybe December: Make this handle all the Pokemon and not just our favorites
p1a_indices = {'Aggron': 0, 'Arceus': 1, 'Cacturne': 2, 'Dragonite': 3, 'Druddigon': 4, 'Uxie': 5}  #p1a is our AI, BloviatingBob
p2a_indices = {'Houndoom': 0, 'Ledian': 1, 'Lugia': 2, 'Malamar': 3, 'Swellow': 4, 'Victreebel': 5} #right now, p2a is a mechanical Turk
combinedIndices = {'Aggron': 809*2+0, 'Arceus': 809*2+1, 'Cacturne': 809*2+2, 'Dragonite': 809*2+3, 'Druddigon': 809*2+4, 'Uxie': 809*2+5,    'Houndoom': 809*2+6, 'Ledian': 809*2+7, 'Lugia': 809*2+8, 'Malamar': 809*2+9, 'Swellow': 809*2+10, 'Victreebel': 809*2+11}

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
                name = temp[2][5:]
                index = pokedex[name.lower()]['num']
                # Not a very efficient solution
                for pokemon in p1a_indices:
                    newIndex = pokedex[pokemon.lower()]['num']
                    if newIndex != index:
                        retval.append([newIndex-1, 0])
                retval.append([index-1, 1])
                index = p1a_indices[name]
                health = int(temp[4].split('/')[0])
                retval.append([809*2 + index, health])
            else:
                assert('p2a' in line)
                temp = line.split('|')
                name = temp[2][5:]
                index = pokedex[name.lower()]['num']
                # Not a very efficient solution
                for pokemon in p2a_indices:
                    newIndex = pokedex[pokemon.lower()]['num']
                    if newIndex != index:
                        retval.append([newIndex-1+809, 0])
                retval.append([index-1+809, 1])
                index = p2a_indices[name]
                health = int(temp[4].split('/')[0])
                retval.append([809*2 + 6 + index, health])
        elif 'damage' in line:
            if 'Substitute' not in line:
                temp = line.split('|')
                name = temp[2][5:]
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
