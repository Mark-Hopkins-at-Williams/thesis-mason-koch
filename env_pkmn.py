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
        # Wait a bit for the game to initialise.
        time.sleep(0.01)
        # It makes sense to wait until turn just for the first time
        retval = ""
        temp = "."
        # TODO soon: delete this and just call scrape_input
        while ("|turn|" not in temp):
            if (temp == ""):
                break
            temp = self.proc.readline().decode()
            retval += temp
        while ("DEADBEEF" not in temp):
            if (temp == ""):
                break
            temp = self.proc.readline().decode()
            retval += temp

        print(retval)
        print(".")
        return retval
    def step(self, action):
        # Obscure method of writing to user input.
        self.proc.sendline(action)
        # Wait for the Pokemon simulator to do its thing. This is not very efficient
        # but it does allow human input.
        time.sleep(0.01)
        return self.scrape_input(), 42.0, self.done, "NotUsed"
    def scrape_input(self):
        retval = ""
        temp = "."
        while ("DEADBEEF" not in temp):
            if (temp == ""):
                # Either the game is over, or there has been an error
                # Regardless,
                self.done = True
                break
            temp = self.proc.readline().decode()
            retval += temp
        #print(retval)
        #print(". end of scrape.")
        return retval
    def pokemon_wrapper(self):
       self.proc = pexpect.spawn("node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js")

