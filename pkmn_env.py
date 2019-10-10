import asyncio
import time
import threading

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
        # source: https://stackoverflow.com/questions/53898231/integer-file-descriptor-0-in-open-python-3/53898574#53898574
        # Turns out you can write directly to standard input. This is very useful since
        # the program given relies heavily on standard input. In the future it might
        # be a good idea to not open and close this so often. Premature optimisation is
        # the root of all evil.
        stin=open(0,"w")
        stin.write(action)
        stin.close()
        # Wait for the Pokemon simulator to do its thing. Inefficient, premature optimisation
        # is the root of all evil.
        time.sleep(1)
        return self.scrape_input(), 42, self.done, "NotUsed"
    def scrape_input(self):
        with self.lock:
            retval = self.observationBuffer
            self.observationBuffer = ""
        return retval
    def pokemon_wrapper(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run('node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js'))
    async def run(self, cmd):
        # Call Pokemon simulator
        self.proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
        self.done = False
        # Sit around waiting for input, exit when the subprocess exits
        data = "."
        while data != "":
            # Notice that if we did not await the readline here, this thread
            # would loop endlessly and consume all the CPU
            data = await self.proc.stdout.readline()
            data = data.decode('ascii').rstrip()
            with self.lock:
                self.observationBuffer += data
        await self.proc.wait()
        self.done = True

