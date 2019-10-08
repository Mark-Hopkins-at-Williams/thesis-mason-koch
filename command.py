import asyncio

async def run(cmd):
    # Call Pokemon simulator
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
    # Sit around waiting for input, exit when the subprocess exits
    data = "."
    while data != "":
        data = await proc.stdout.readline()
        data = data.decode('ascii').rstrip()
        # Print the data to the screen so the user can see it
        print(data)
    await proc.wait()

# Run the random player
asyncio.run(run('node ./Pokemon-Showdown/.sim-dist/examples/test_random_player.js'))

