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
            running = True
            while (running):
                simulator_response = self.proc.readline().decode()
                simulator_responses = simulator_response.split("|")
                for s in simulator_responses:
                    if "error" in s:
                        for __ in range(7):  # I don't remember why this is 7.
                            simulator_response += self.proc.readline().decode()
                        raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + simulator_response)
                    elif 'actionspace' in s:
                        if len(s) > 24 and s[24] == '[':
                            self.action_space = json.loads(s[24:])
                        else:
                            self.action_space = json.loads(s[11:])
                    elif 'opponentspace' in s:
                        self.opponent_action_space = json.loads(s[13:])
                    elif "gameinfo" in s:
                        retval += s[8:]
                    elif "HughMann" in s:
                        if "HughMannyes" in s:
                            self.reward = 1.0
                        else:
                            assert('HughMannno' in s)
                            self.reward = -1.0
                        self.done = True
                        return retval
                    elif "DEADBEEF" in s:
                        running = False
        return retval

