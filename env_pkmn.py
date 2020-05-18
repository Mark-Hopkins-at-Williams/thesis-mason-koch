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
        self.action_space_mapping = {}
        self.opponent_action_space_mapping = {}
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
        self.i = 0.0
        self.proc.sendline(start_command)
        return self.scrape_input()
    def step(self, action):
        # In normal scenarios, the simulator always gives us four moves. However when
        # we get a two-turn move, it only gives us one move. If you want to use that
        # move (and if you don't want to use it, the game will crash), you must enter
        # "move 1" regardless of what slot that move was originally in. This is a hack
        # to get around this.
        actions = action.split("|")
        if len(actions[0]) > 0 and actions[0][0] == 'm' and actions[0] != "move struggle":
            actions[0] = self.opponent_action_space_mapping[actions[0]]
        if len(actions[1]) > 0 and actions[1][0] == 'm' and actions[1] != "move struggle":
            actions[1] = self.action_space_mapping[actions[1]]
        action = actions[0] + "|" + actions[1]
        self.proc.sendline(action)
        return self.scrape_input(), self.reward, self.done, "NotUsed"
    def check_spaces_for_validity(self):
        # Hack to get around "move 1" not being the same move when we are using a two-turn move
        self.action_space_mapping = {}
        self.opponent_action_space_mapping = {}
        if "move struggle" not in self.action_space:
            for i in range(len(self.action_space)):
                if self.action_space[i][0] == 'm':
                    self.action_space_mapping["move " + self.action_space[i][7]] = "move " + self.action_space[i][5]
                    self.action_space[i] = "move " + self.action_space[i][7]
        if "move struggle" not in self.opponent_action_space:
            for i in range(len(self.opponent_action_space)):
                if self.opponent_action_space[i][0] == 'm':
                    self.opponent_action_space_mapping["move " + self.opponent_action_space[i][7]] = "move " + self.opponent_action_space[i][5]
                    self.opponent_action_space[i] = "move " + self.opponent_action_space[i][7]
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
                    elif self.i > 300:
                        # Kill games which last more than 300 turns. On the web, the limit is 1000 turns, but games frequently end by mutual agreement well before that.
                        self.proc.close(force = True)
                        self.reward = -1.0
                        self.done = True
                        return retval
                    elif "DEADBEEF" in s:
                        running = False
        self.check_spaces_for_validity()
        self.i += 1
        return retval

