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
        self.proc = pexpect.spawn("python run.py")
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
            if ('|win|' in temp):
                # Either the game is over, or there has been an error
                # Regardless,
                print(temp)
                self.done = True
                if "|win|BloviatingBob" in retval:
                    self.reward = 1.0
                else:
                    self.reward = -1.0
                break
            # Action space not implemented for the smogon version.
            temp = self.proc.readline().decode()
            retval += temp
        if "error" in retval:
            raise Exception("The Pokemon simulator crashed. The most recent communication from it was:\n" + retval)
        return retval

