import asyncio
import time
import websockets
from websocket_client import PSWebsocketClient as Client

async def main():
    myclient = Client()
    newclient = await myclient.create('BloviatingBob', '12345', 'sim.smogon.com:8000')
    time.sleep(0.01)
    await newclient.login()
    time.sleep(0.01)
    await newclient.challenge_user('aiDebugNotABattle', 'gen7customgame', 'Aggron||leftovers|sturdy|headsmash,heavyslam,aquatail,superpower||85,85,85,85,85,85||||100|]Arceus|arceusfighting|fistplate|multitype|swordsdance,extremespeed,aerialace,dracometeor||85,,85,85,85,85|N|,0,,,,||81|')

    roomname = ""
    # Wait until the room gets initialised
    while True:
        msg = await newclient.receive_message()
        # This print statement is not for debugging; it is what env_pokemon_smogon receives
        print(msg)
        if '"games":{"battle-gen7customgame' in msg:
            # Hope these messages are always the same length
            roomname = msg[-64:-32]
            break
    # Choose the default Pokemon to start. (You only have 1 choice, so this is always correct).
    # Then make it so other people don't join the chatroom by accident to see really dull Pokemon
    # being played by robots.
    while True:
        msg = await newclient.receive_message()
        # Similarly not for debugging.
        print(msg)
        if 'teampreview' in msg:
            await newclient.send_message(roomname, ['/choose default'])
            await newclient.send_message(roomname, ['/modjoin +'])
            break
    waiting = False
    # TODO: This is probably debugging. Remove it.
    print("ROOMNAME IS: " + roomname)
    while True:
        msg = await newclient.receive_message()
        if 'updatechallenges' in msg:
            # dummy fake reward for debugging. TODO: remove this, since it is likely a relic from when
            # the goal was to run more than one battle with pkmn_smogon.
            print("|win|AmazingAlice")
            msg = "DEADBEEF"
        # Not for debugging
        print(msg)
        if 'request' in msg:
            if msg[11:15] == 'wait':
                waiting = True
            else:
                if msg[11:17] == 'active':
                    waiting = False
        if msg == "DEADBEEF":
            if not waiting:
                # This is how env_smogon communicates with this file
                inpt = input()
                await newclient.send_message(roomname, ['/' + inpt])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

