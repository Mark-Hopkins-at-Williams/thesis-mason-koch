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

