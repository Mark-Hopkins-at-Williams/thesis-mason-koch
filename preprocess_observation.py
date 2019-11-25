""" preprocess_observation for pkmn.py. Kept in its own separate file in part because it is long. Functions like this generally exist for all games, but they differ a lot game to game. """
import json
from pokedex import pokedex


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
        health = int(pokemon['condition'].split('/')[0].split(' ')[0])
        retval.append([809*2 + tempDict[name], health])

        index = pokedex[name.lower()]['num']
        if pokemon['active']:
            retval.append([index-1, 1])
        else:
            retval.append([index-1, 0])
        retval2[tempDict[name]] = k
        k += 1
    other_indices = [pokedex['houndoom']['num'], pokedex['arceus']['num'], pokedex['cacturne']['num'], pokedex['dragonite']['num'], pokedex['druddigon']['num'], pokedex['uxie']['num']]
    assert(int(mydict['State'][6]) != 6)  # Remove this eventually
    cur_pokemon = int(mydict['State'][6])
    for i in [0,1,2,3,4,5]:
        # Append health information
        retval.append([809*2+6+i, int(mydict['State'][i].split('/')[0].split(' ')[0])])
        # Append which Pokemon is on the field
        retval.append([809-1+ other_indices[i], i == cur_pokemon])

    #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    return retval, retval2

