import pexpect
import json

class Env():
    def __init__(self):
        self.done = True
        self.action_space = []
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
        temp = "."
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
            # The action space line will still get transferred in the response.
            # There's no good reason for this beyond ''well it can't hurt''.
            if 'actionspace' in temp:
                temp2 = json.loads(temp[11:])[0]
                self.action_space = temp2[1]
                for temp3 in temp2[0]:
                    self.action_space.append(temp3['choice'])
            temp = self.proc.readline().decode()
            retval += temp
        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

