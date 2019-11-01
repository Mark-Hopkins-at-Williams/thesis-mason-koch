import time
import threading
import os
import subprocess
import pexpect

class Env():
    def __init__(self):
        self.done = True
    def seed(self, num):
        dummy = True
    def render(self):
        raise NotImplementedError()
    def reset(self):
        # Create a Pokemon battle.
        self.t = threading.Thread(target = self.pokemon_wrapper)
        self.t.start()
        self.done = False
        self.reward = 0.0
        # Wait a bit for the game to initialise.
        time.sleep(0.01)
        return self.scrape_input()
    def step(self, action):
        # Obscure method of writing to user input.
        self.proc.sendline(action)
        # Wait for the Pokemon simulator to do its thing. This is not very efficient
        # but it does allow human input if you make it sleep a little longer.
        time.sleep(0.01)
        return self.scrape_input(), self.reward, self.done, "NotUsed"
    def scrape_input(self):
        retval = ""
        temp = "."
        while ("DEADBEEF" not in temp):
            if (temp == ""):
                # Either the game is over, or there has been an error
                # Regardless,
                self.done = True
                if "|win|HughMann" in retval:
                    self.reward = 1.0
                else:
                    self.reward = -1.0
                break
            temp = self.proc.readline().decode()
            retval += temp
        return retval
    def pokemon_wrapper(self):
       self.proc = pexpect.spawn("node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js")

