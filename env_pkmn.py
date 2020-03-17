import pexpect
import json
# Testing indicates numpy's random number generator, which is used elsewhere, is distinct from Python's.
# So this works. It's not the most elegant, since anyone using random may be in for some nasty surprises when their random number generator gets called when it is not supposed to.
# To the best of my knowledge, Python doesn't have any random number generator objects.
# I could run the random number generator in a subprocess. But that is a total hack, so I am going to just let people read these comments instead.
import random 
from game_model import FNAME

class Env():
    def __init__(self):
        self.done = True
        self.action_space = []
        self.opponent_action_space = []
    def seed(self, num):
        random.seed(num)
    def render(self):
        raise NotImplementedError()
    def reset(self, start_command):
        # Create a Pokemon battle. The random number seed ultimately finds its way to prng in the Pokemon-Showdown/sim directory.
        self.proc = pexpect.spawn("node ./Pokemon-Showdown/.sim-dist/examples/" + FNAME + ".js [" + str(random.randint(0, 65535)) + "," + str(random.randint(0, 65535)) + "," +str(random.randint(0, 65535)) + "," +str(random.randint(0, 65535)) + "]")
        self.proc.delaybeforesend = 0.001  # Very thankful to https://stackoverflow.com/questions/60215395/pexpect-sendline-is-too-slow on this one.
        self.done = False
        self.reward = 0.0
        self.proc.sendline(start_command)
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
                elif "HughMann" in simulator_response and "|player|p2|HughMann||" not in simulator_response:
                    retval += simulator_response
                simulator_response = self.proc.readline().decode()
            if simulator_response == "":
                break

        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

