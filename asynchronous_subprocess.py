import asyncio
import time
import websockets
from websocket_client import PSWebsocketClient as Client

async def main():
    myclient = Client()
    # In a better world we could just say client = Client(). However the create method needs to be asynchronous,
    # and awaiting the creation of a class object seems weird.
    client = await myclient.create('BloviatingBob', '12345', 'sim.smogon.com:8000')
    time.sleep(0.01)
    await client.login()
    time.sleep(0.01)
    await client.challenge_user('aiDebugNotABattle', 'gen7customgame', 'Swellow||flameorb|guts|bravebird,earthquake,swordsdance,facade||85,85,85,85,85,85||||100|]Ledian||leftovers|swarm|toxic,stealthrock,uturn,roost||85,85,85,85,85,85||||100|')
    # remember the room name so we can send messages to it
    roomname = ""
    # Wait until the room gets initialised
    while True:
        msg = await client.receive_message()
        # Note that every print statement in this file is not for debugging; it is what env_pokemon_smogon receives
        print(msg)
        if '"games":{"battle-gen7customgame' in msg:
            # Hope these messages are always the same length
            roomname = msg[-64:-32]
            break
    # Choose the default Pokemon to start. Then make it so other people don't join
    # the chatroom by accident to see really dull Pokemon being played by robots.
    while True:
        msg = await client.receive_message()
        print(msg)
        if 'teampreview' in msg:
            await client.send_message(roomname, ['/choose default'])
            await client.send_message(roomname, ['/modjoin +'])
            break
    waiting = False
    while True:
        msg = await client.receive_message()
        if 'updatechallenges' in msg:
            raise Exception("I don't remember the edge case this covered. Previously it was handled by printing '|win|AmazingAlice' and setting message to DEADBEEF.")
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
                await client.send_message(roomname, ['/' + inpt])

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

