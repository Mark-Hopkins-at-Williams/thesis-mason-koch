import asyncio

async def run(cmd):
    # Call "Pokemon simulator"
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
    # Sit around waiting for input and exit when the subprocess exits
    data = "."
    while data != "":
        data = await proc.stdout.readline()
        data = data.decode('ascii').rstrip()
        # Print the data to the screen, because why not?
        print(data)
    await proc.wait()

# In this case, we are running a simple program which returns what we fed it plus a little more
asyncio.run(run('node hello.js'))

