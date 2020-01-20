import pexpect
from pokedex import pokedex
import game_model as gm
from game_factory import GameFactory

class Env():
    def __init__(self):
        self.done = True
        self.action_space = []
    def seed(self, num):
        pass
    def render(self):
        raise NotImplementedError()
    def reset(self):
        # Create a Pokemon battle.
        self.proc = pexpect.spawn("python asynchronous_subprocess.py")
        self.done = False
        self.reward = 0.0
        return self.scrape_input()
    def step(self, action):
        self.proc.sendline(action)
        return self.scrape_input(), self.reward, self.done, "NotUsed"
    def scrape_input(self):
        retval = ""
        simulator_response = "."
        while ("DEADBEEF" not in simulator_response):
            if ('|win|' in simulator_response):
                # Either the game is over, or there has been an error
                # Regardless,
                print(simulator_response)
                self.done = True
                break
            # Action space not implemented for the smogon version.
            simulator_response = self.proc.readline().decode()
            retval += simulator_response
        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval



p1a_indices = {}
p2a_indices = {}
combinedIndices = {}
for i in range(6):
    p1a_indices[gm.OUR_TEAM[i]] = i
    p1a_indices[i] = pokedex[gm.OUR_TEAM[i]]['num']
    p2a_indices[gm.OPPONENT_TEAM[i]] = i
    p2a_indices[i] = pokedex[gm.OPPONENT_TEAM[i]]['num']
    combinedIndices[gm.OUR_TEAM[i]] = gm.NUM_POKEMON*2+i
    combinedIndices[gm.OPPONENT_TEAM[i]] = gm.NUM_POKEMON*2+i+6

def preprocess_observation(I):
    # In this case, the string we get back from the Pokemon simulator does not
    # give us the entire state of the game. Instead it gives us the change in
    # the state. So return a list. Each element of the list contains two
    # elements, the index of the state to update and the value to update it to.
    retval = []
    I = I.split('\n')
    for line in I:
        split_line = line.split('|')
        if ('switch|' in line) or ('drag|' in line):
            # There is a new Pokemon on the field.
            if 'p1a' in line:
                # This relevant_indices solution is not markedly more elegant
                # than what it replaced. In that respect, despite the rewrite,
                # this section is still unsatisfying.
                # It should be easier to maintain, however.
                relevant_indices = p1a_indices
                relevant_offsets = [0,0,0]
            else:
                assert('p2a' in line)
                relevant_indices = p2a_indices
                relevant_offsets = [gm.NUM_POKEMON, 
                                    gm.TEAM_SIZE, 
                                    gm.NUM_STATUS_CONDITIONS*gm.TEAM_SIZE]
            # Update the pokemon on field
            name = split_line[2][5:].lower()
            index = pokedex[name]['num']
            for i in range(6):
                retval.append([relevant_indices[i]-1 + relevant_offsets[0], 
                               relevant_indices[i] == index])
            # And the health
            condition = split_line[4].split('/')[0].split(' ')
            health = int(condition[0])
            retval.append([gm.OFFSET_HEALTH + relevant_offsets[1] + 
                           relevant_indices[name], health])
            # And the status conditions (or lack thereof)
            if len(condition) != 1:
                assert(len(condition) == 2)
                for i in range(gm.NSC_PLACEHOLDER):
                    retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                                   relevant_offsets[2] + 
                                   gm.NUM_STATUS_CONDITIONS * 
                                   relevant_indices[name] + i, 
                                   gm.STATUS_DICT[condition[1]] == i])
            else:
                for i in range(gm.NSC_PLACEHOLDER):
                    retval.append([gm.OFFSET_STATUS_CONDITIONS + 
                                   relevant_offsets[2] + 
                                   gm.NUM_STATUS_CONDITIONS * 
                                   relevant_indices[name] + i, 
                                   0])
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
            retval.append(gm.OFFSET_STAT_BOOSTS + 
                          ('p2a' in line)*gm.NUM_STAT_BOOSTS + 
                          gm.BOOST_DICT[split_line[3]], 
                          -1 * float(split_line[4]))
        elif 'boost|' in line:
            if 'Swarm' not in line:
                name = split_line[2][5:].lower()
                retval.append([gm.OFFSET_STAT_BOOSTS + ('p2a' in line)*
                               gm.NUM_STAT_BOOSTS + gm.BOOST_DICT[split_line[3]], 
                               float(split_line[4])])
        elif 'weather|' in line:
            if 'upkeep' in line:
                # the weather has stopped
                retval.append([gm.OFFSET_WEATHER, 1])
                for i in range(1, gm.NUM_WEATHER):
                    retval.append([gm.OFFSET_WEATHER + i, 0])
            else:
                # The weather has started
                retval.append([gm.OFFSET_WEATHER, 0])
                for i in range(1, gm.NUM_WEATHER):
                    retval.append([gm.OFFSET_WEATHER + i, 
                                   i == gm.WEATHER_DICT[split_line[2][:-1].lower()]])
        elif 'fieldstart|' in line:
            retval.append([gm.OFFSET_TERRAIN +0, 0])
            for i in range(1, gm.NUM_TERRAIN):
                retval.append([gm.OFFSET_TERRAIN+i, 
                               gm.TERRAIN_LOOKUP[split_line[2]] == i])
        elif 'fieldend|' in line:
            retval.append([gm.OFFSET_TERRAIN, 1])
            for i in range(1,gm.NUM_TERRAIN):
                retval.append([gm.OFFSET_TERRAIN+i, 0])
        elif 'sidestart|' in line:
            hazard = split_line[-1][:-1].lower()
            retval.append([gm.OFFSET_HAZARDS + gm.HAZARD_DICT[hazard] + 
                           gm.NUM_HAZARDS * ('p2' in line), 1])
        elif 'sideend|' in line:
            hazard = split_line[-1][:-1].lower()
            retval.append([gm.OFFSET_HAZARDS + gm.HAZARD_DICT[hazard] + 
                           gm.NUM_HAZARDS * ('p2' in line), 0])

    print(retval)
    return retval

class SmogonGameFactory(GameFactory):
    
    def __init__(self):
        pass

    def create_env(self):
        return Env()
        
    def create_observation_preprocessor(self):
        return preprocess_observation
    
    