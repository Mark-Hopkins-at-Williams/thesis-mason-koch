import pexpect
import json
from pokedex import pokedex
import game_model as gm


class GameFactory:
    
    def create_env(self):
        raise NotImplementedError("Cannot call .create_env on a base class.")
        
    def create_observation_preprocessor(self):
        raise NotImplementedError("Cannot call .create_env on a base class.")
        

class BasicEnv():
    def __init__(self):
        self.done = True
        self.action_space = []
        self.opponent_action_space = []
    def seed(self, num):
        raise NotImplementedError()
    def render(self):
        raise NotImplementedError()
    def reset(self):
        # Create a Pokemon battle.
        self.proc = pexpect.spawn("node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js")
        self.done = False
        self.reward = 0.0
        return self.scrape_input()
    def step(self, action):
        self.proc.sendline(action)
        return self.scrape_input(), self.reward, self.done, "NotUsed"
    def scrape_input(self):
        retval = ""
        # Wait until both the AI we are training and the other one get back to us
        for _ in range (2):
            simulator_response = self.proc.readline().decode()
            while ("DEADBEEF" not in simulator_response):
                if (simulator_response == ""):
                    # Either the game is over, or there has been an error
                    # Regardless,
                    self.done = True
                    if "HughMannyes" in retval:
                        self.reward = 1.0
                    else:
                        assert('HughMannno' in retval)
                        self.reward = -1.0
                    break
                elif 'actionspace' in simulator_response:
                    loaded_JSON = ''
                    # There's some chicanery going around in simulator_response. This is a workaround.
                    if len(simulator_response) > 24 and simulator_response[24] == '[':
                        loaded_JSON = json.loads(simulator_response[24:])[0]
                    else:
                        loaded_JSON = json.loads(simulator_response[11:])[0]
                    if len(loaded_JSON) > 0:
                        self.action_space = loaded_JSON[1]
                        for move in loaded_JSON[0]:
                            self.action_space.append(move['choice'])
                    else:
                        self.action_space = []
                elif 'opponentspace' in simulator_response:
                    loaded_JSON = json.loads(simulator_response[13:])[0]
                    if len(loaded_JSON) > 0:
                        self.opponent_action_space = loaded_JSON[1]
                        for move in loaded_JSON[0]:
                            self.opponent_action_space.append(move['choice'])
                    else:
                        self.opponent_action_space = []
                elif "gameinfo" in simulator_response:
                    retval += simulator_response[8:]
                elif "HughMann" in simulator_response:
                    retval += simulator_response
                simulator_response = self.proc.readline().decode()
            if simulator_response == "":
                break

        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

def basic_preprocess_observation(I):
    # The argument to this function is a stringified JSON object which
    # ultimately hails from battle-stream in the Pokemon-Showdown/sim directory.
    mydict = json.loads(I)
    retval = []
    for pokemon in mydict['side']['pokemon']:
        name = pokemon['ident'][4:]
        ordinal_index = gm.OUR_TEAM[name.lower()]
        # Add health information for this pokemon
        condition = pokemon['condition'].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([gm.OFFSET_HEALTH + ordinal_index, health])
        # Add status information for this Pokemon
        if len(condition) != 1 and condition[1] != 'fnt':
            assert(len(condition) == 2)
            for i in range(7):
                retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                               gm.NUM_STATUS_CONDITIONS*ordinal_index + i, 
                               gm.STATUS_DICT[condition[1]] == i])
        else:
            for i in range(7):
                retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                               gm.NUM_STATUS_CONDITIONS*ordinal_index  + i, 0])
        # Add whether this Pokemon is active
        index = pokedex[name.lower()]['num']
        retval.append([index-1, pokemon['active']])

    active_pokemon = int(mydict['State'][6])
    for i in range(6):
        # Append health information. This and the status information is almost
        # identical to that for the Pokemon on our side.
        condition = mydict['State'][i].split('/')[0].split(' ')
        health = int(condition[0])
        retval.append([gm.OFFSET_HEALTH+gm.TEAM_SIZE + i, health])
        # Add status information
        if len(condition) != 1 and condition[1] != 'fnt':
            assert(len(condition) == 2)
            for j in range(7):
                retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                               gm.NUM_STATUS_CONDITIONS * 
                               (gm.TEAM_SIZE + i) + j, 
                               gm.STATUS_DICT[condition[1]] == j])
        else:
            for j in range(7):
                retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                               gm.NUM_STATUS_CONDITIONS * 
                               (gm.TEAM_SIZE + i) + j, 0])
        # Append which Pokemon is on the field
        index = pokedex[gm.OPPONENT_TEAM[i]]['num']
        retval.append([gm.NUM_POKEMON + index-1, i == active_pokemon])

    # Deal with 7 stat boosts on each team
    for i in [7,8,9,10,11,12,13, 14,15,16,17,18,19,20]:
        retval.append([gm.OFFSET_STAT_BOOSTS + i-7, mydict['State'][i]])
    # Add weather
    for i in range(6):
        retval.append([gm.OFFSET_WEATHER + i, 
                       i == gm.WEATHER_DICT[mydict['State'][21]]])
    # And terrain
    for i in range(4):
        retval.append([gm.OFFSET_TERRAIN + i, 
                       i == gm.TERRAIN_DICT[mydict['State'][22]]])
    # And entry hazards! Note that these are values like ['spikes', 'toxicspikes'].
    # Crucially, 'spikes' is not in ['toxicspikes'].
    for hazard, index in gm.HAZARD_DICT.items():
        retval.append([gm.OFFSET_HAZARDS+index, 
                       hazard in mydict['State'][23]])
        retval.append([gm.OFFSET_HAZARDS+index+4, 
                       hazard in mydict['State'][24]])
    # We are returning a list of index-value pairs, where you look in to
    # the state and replace the thing at the index with the given value.
    return retval


       
class BasicGameFactory(GameFactory):
    
    def __init__(self):
        pass

    def create_env(self):
        return BasicEnv()
        
    def create_observation_preprocessor(self):
        return basic_preprocess_observation
    
    