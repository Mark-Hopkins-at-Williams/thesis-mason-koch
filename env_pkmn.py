import pexpect
import json

class Env():
    def __init__(self):
        self.done = True
        self.action_space = []
        self.opponent_space = []
    def seed(self, num):
        dummy = True
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
        for UNUSED in range (2):
            temp = self.proc.readline().decode()
            while ("DEADBEEF" not in temp):
                if (temp == ""):
                    # Either the game is over, or there has been an error
                    # Regardless,
                    self.done = True
                    if "HughMannyes" in retval:
                        self.reward = 1.0
                    else:
                        assert('HughMannno' in retval)
                        self.reward = -1.0
                    break
                elif 'actionspace' in temp:
                    temp2 = ''
                    # There's some chicanery going around in temp. This is a workaround.
                    if len(temp) > 24 and temp[24] == '[':
                        temp2 = json.loads(temp[24:])[0]
                    else:
                        temp2 = json.loads(temp[11:])[0]
                    if len(temp2) > 0:
                        self.action_space = temp2[1]
                        for temp3 in temp2[0]:
                            self.action_space.append(temp3['choice'])
                    else:
                        self.action_space = []
                elif 'opponentspace' in temp:
                    temp2 = json.loads(temp[13:])[0]
                    if len(temp2) > 0:
                        self.opponent_space = temp2[1]
                        for temp3 in temp2[0]:
                            self.opponent_space.append(temp3['choice'])
                    else:
                        self.opponent_space = []
                elif "gameinfo" in temp:
                    retval += temp[8:]
                elif "HughMann" in temp:
                    retval += temp
                temp = self.proc.readline().decode()
            if temp == "":
                break

        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

