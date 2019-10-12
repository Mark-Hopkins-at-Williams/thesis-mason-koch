import asyncio
import time
import threading
import os

class Env():
    def __init__(self):
        self.proc = ""
        self.done = True
        self.lock = ""
        self.observationBuffer = ""
    def seed(self, num):
        dummy = True
    def render(self):
        print(notDefinedYet)
    def reset(self):
        # Create a Pokemon battle in a separate thread.
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        asyncio.get_child_watcher().attach_loop(self.loop)
        self.t = threading.Thread(target = self.pokemon_wrapper, args = (self.loop,))
        self.t.start()
        # Wait a bit for the game to initialise.
        time.sleep(2)
        # The Pokemon battle will return some stuff
        return self.scrape_input()
    def step(self, action):
        # I'm trying to write to standard input in various ways here. This is because if
        # you type in to standard input from the keyboard, it works.

        # Attempt 1, open file descriptor 0 (which is standard input).
        # Gives no errors, also doesn't work.
        #stin=open(0,"w")
        #stin.write(action + "\n")
        #stin.close()
        
        # Attempt 2, tell the operating system to print the action.
        # This produces no errors, but it also doesn't work.
        #os.system("echo " + action)

        # Attempt 3, communicate with the process directly.
        # Communicate, like all of process's methods, is asynchronous, so this gives an error.
        #self.proc.communicate(action.encode())
        # This gives a "loop is being detached from a child watcher with pending handlers" warning
        # and an "attached to a different loop" error. (this attachment is probably in run).
        #asyncio.run(self.proc.communicate(action.encode()))

        # Attempt 4, communicate with the process directly in a different thread.
        # This also gives an attached to a different loop error.
        #myloop = asyncio.new_event_loop()
        #mythread = threading.Thread(target = self.wr, args = (myloop, action,))
        #mythread.start()

        # Wait for the Pokemon simulator to do its thing. This is not very efficient
        # but it does allow human input (which works).
        time.sleep(3)
        return self.scrape_input(), 42, self.done, "NotUsed"
    def wr(self, myloop, action):
        # Used in the third attempt to write to the Pokemon simulator.
        myloop.run_until_complete(self.proc.communicate(action.encode))
    def scrape_input(self):
        with self.lock:
            retval = self.observationBuffer
            self.observationBuffer = ""
        return retval
    def pokemon_wrapper(self, loop):
        asyncio.set_event_loop(loop)
        # This really will run until complete, which is why this is in a separate thread.
        loop.run_until_complete(self.run('node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js'))
    async def run(self, cmd):
        # Call Pokemon simulaton
        self.proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
        #self.proc = await asyncio.create_subprocess_shell(cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        # The above seems like the way to go; it does indeed prevent user input from doing anything.
        # But attempting to write to the subprocess gives the same issues as before.
        self.done = False
        # Sit around waiting for input, exit when the subprocess exits
        data = "."
        while data != "":
            # Notice that if we did not await the readline here, this thread
            # would loop endlessly and consume all the CPU
            data = await self.proc.stdout.readline()
            #data = await self.proc.communicate(None) # doesn't work
            data = data.decode('ascii').rstrip()
            with self.lock:
                self.observationBuffer += data
            # Print all the data we receive for debugging purposes
            print(data)
        await self.proc.wait()
        self.done = True

