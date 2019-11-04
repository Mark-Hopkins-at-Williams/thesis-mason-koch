""" preprocess_observation for pkmn.py. Kept in its own separate file in part because it is long. Functions like this generally exist for all games, but they differ a lot game to game. """
import numpy as np # I sure remember importing something in both a file and its dependent file being a bad idea. Added to the end of the todo list.

def preprocess_observation(I):
    # There will always be a preprocessing step, but it will look different for different games.
    # In this case, the string we get back from the Pokemon simulator does not give us the entire state
    # of the game. Instead it gives us the change in the state. So return a list.
    # Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    retval = []
    I = I.split('\n')
    dbi = -1
    ci = 0
    retval2 = [-1,-1,-1,-1,-1,-1]
    for line in I:
        if ('switch|' in line) or ('drag|' in line):
            # There is a new Pokemon on the field. Update the pokemon on field and the health.
            if 'p1a' in line:
                temp = line.split('|')
                # Todo in maybe December: Make this handle all the Pokemon and not just our favorites
                tempDict = {'Houndoom': 0, 'Ledian': 1, 'Lugia': 2, 'Malamar': 3, 'Swellow': 4, 'Victreebel': 5}
                name = temp[2][5:]
                index = tempDict[name]
                retval.append([0, index])
                health = int(temp[-1].split('/')[0])
                retval.append([2 + index, health])
            else:
                assert('p2a' in line)
                temp = line.split('|')
                tempDict = {'Aggron': 0, 'Arceus': 1, 'Cacturne': 2, 'Dragonite': 3, 'Druddigon': 4, 'Uxie': 5}
                name = temp[2][5:]
                index = tempDict[name]
                retval.append([1, index])
                health = int(temp[-1].split('/')[0])
                retval.append([8 + index, health])
        elif 'damage' in line:
            if 'Substitute' not in line:
                tempDict = {'Houndoom': 2, 'Ledian': 3, 'Lugia': 4, 'Malamar': 5, 'Swellow': 6, 'Victreebel': 7, 'Aggron': 8, 'Arceus': 9, 'Cacturne': 10, 'Dragonite': 11, 'Druddigon': 12, 'Uxie': 13}
                temp = line.split('|')
                name = temp[2][5:]
                if temp[-1][0] == '[':
                    #The simulator is telling us the source of the damage
                    health = 0
                    if 'fnt' not in temp[-2]:
                        health = int(temp[-2].split('/')[0])
                    retval.append([tempDict[name], health])
                else:
                    if 'fnt' in temp[-1]:
                        health = 0
                        retval.append([tempDict[name], health])
                    else:
                        health = int(temp[-1].split('/')[0])
                        retval.append([tempDict[name], health])
        elif 'DEADBEEF' in line:
            dbi = ci
        elif line == 'p2: Aggron\r':
            retval2[0] = ci
        elif line == 'p2: Arceus\r':
            retval2[1] = ci
        elif line == 'p2: Cacturne\r':
            retval2[2] = ci
        elif line == 'p2: Dragonite\r':
            retval2[3] = ci
        elif line == 'p2: Druddigon\r':
            retval2[4] = ci
        elif line == 'p2: Uxie\r':
            retval2[5] = ci
        ci += 1

    #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    retval2 -= np.min(retval2)
    retval2 += 1
    return retval, retval2
