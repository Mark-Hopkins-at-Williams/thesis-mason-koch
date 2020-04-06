import pexpect
import json
from game_model import *

class Env():
    def __init__(self):
        self.done = True
        self.action_space = [False for i in range(10)]
        self.action_space_mapping = {}
    def seed(self, num):
        dummy = True
    def render(self):
        raise NotImplementedError()
    def reset(self):
        # Create a Pokemon battle.
        self.proc = pexpect.spawn("python asynchronous_subprocess.py")
        self.done = False
        self.reward = 0.0
        return self.scrape_input()
    def step(self, action):
        if action[0] == 'm':
            action = self.action_space_mapping[action]
        self.proc.sendline(action)
        return self.scrape_input(), self.reward, self.done, "NotUsed"
    def scrape_input(self):
        retval = ""
        simulator_response = "."
        self.action_space = [False for i in range(10)]
        # INACTIVE|T MIGHT BE LOSING SOMETHING? HOPEFULLY NOT.
        while ("DEADBEEF" not in simulator_response and "|inactive|T" not in simulator_response):
            print(simulator_response)
            if ('|win|' in simulator_response):
                # Either the game is over, or there has been an error
                # Regardless,
                print(simulator_response)
                self.done = True
                break
            if '|request|{"active"' in simulator_response:
                self.action_space_mapping = {}
                # Ugly JSON parsing. TODO: REPLACE THIS WITH ACTUAL JSON?
                index = simulator_response.find("|request")
                index2 = simulator_response.find(",\"side\"")
                available_moves = simulator_response[int(index)+19:int(index2)]
                available_moves = json.loads(available_moves)
                index3 = simulator_response.index("active\":true")
                index4 = simulator_response[index3:].index("\"moves\"")
                index5 = simulator_response[index3+index4:].index("]")
                all_moves = simulator_response[index3+index4+8:index3+index4+index5+1]
                all_moves = json.loads(all_moves)
                assert(len(all_moves)== 4)
                # Assume the move is not usable
                for i in range(4):
                    self.action_space[i] = True
                for i in range(4):
                    j = 0
                    for move in available_moves[0]['moves']:
                        if "hiddenpower" in move['id']:
                            move['id'] = "hiddenpower"
                        if all_moves[i] == move['id']:
                            # Then we have matched the move to the correct index.
                            self.action_space_mapping["move " + str(i+1)] = "move " + str(j+1)
                            if "disabled" in move and move["disabled"]:
                                self.action_space[i] = True
                            else:
                                # Otherwise, this move is fine
                                self.action_space[i] = False
                        j += 1
                for i in range(4, 10):
                    self.action_space[i] = "trapped" in simulator_response
            # Action space implemented differently for the smogon version.
            simulator_response = self.proc.readline().decode()
            retval += simulator_response
            if '|request|{"wait"' in simulator_response:
                # Wait for our opponent to make a move
                self.action_space = []
        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

