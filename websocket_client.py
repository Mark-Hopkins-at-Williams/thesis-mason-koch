# Ripped from pmariglia's github at https://github.com/pmariglia/showdown and subsequently changed to meet my needs.
import asyncio
import websockets
import requests
import json
import time


class LoginError(Exception):
    pass


class SaveReplayError(Exception):
    pass


class PSWebsocketClient:

    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    last_message = None
    last_challenge_time = 0

    @classmethod
    async def create(cls, username, password, address):
        self = PSWebsocketClient()
        self.username = username
        self.password = password
        self.address = "ws://{}/showdown/websocket".format(address)
        self.websocket = await websockets.connect(self.address)
        self.login_uri = "https://play.pokemonshowdown.com/action.php"
        return self

    async def receive_message(self):
        try:
            # pmariglia's AI, which is what we are fighting against, takes an extremely long time
            # to decide what Pokemon to put on the field.
            message = await asyncio.wait_for(self.websocket.recv(), timeout = 18)
            return message
        except asyncio.TimeoutError:
            # The timeout from the server is our chosen method of determining when we need to make a move. 
            # This is extremely slow, and chosen mostly because it works.
            return "DEADBEEF"

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list)
        await self.websocket.send(message)
        self.last_message = message
    # TODO: delete this function since I don't think it ever gets used
    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            split_message = message.split('|')
            if split_message[1] == 'challstr':
                return split_message[2], split_message[3]

    async def login(self):
        client_id, challstr = await self.get_id_and_challstr()
        if self.password:
            response = requests.post(
                self.login_uri,
                data={
                    'act': 'login',
                    'name': self.username,
                    'pass': self.password,
                    'challstr': "|".join([client_id, challstr])
                }
            )

        else:
            response = requests.post(
                self.login_uri,
                data={
                    'act': 'getassertion',
                    'userid': self.username,
                    'challstr': '|'.join([client_id, challstr]),
                }
            )

        if response.status_code == 200:
            if self.password:
                response_json = json.loads(response.text[1:])
                assertion = response_json.get('assertion')
            else:
                assertion = response.text

            message = ["/trn " + self.username + ",0," + assertion]
            await self.send_message('', message)
        else:
            raise LoginError("Could not log-in")

    async def update_team(self, team):
        message = ["/utm {}".format(team)]
        await self.send_message('', message)

    async def challenge_user(self, user_to_challenge, battle_format, team):
        if time.time() - self.last_challenge_time < 10:
            await asyncio.sleep(10)
        await self.update_team(team)
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message('', message)
        self.last_challenge_time = time.time()

    async def accept_challenge(self, battle_format, team):
        await self.update_team(team)
        username = None
        while username is None:
            msg = await self.receive_message()
            split_msg = msg.split('|')
            if split_msg[1] == 'updatechallenges':
                try:
                    challenges = json.loads(split_msg[2])
                    if challenges['challengesFrom'] is not None:
                        username, challenge_format = next(iter(challenges['challengesFrom'].items()))
                        if challenge_format != battle_format:
                            username = None
                except ValueError:
                    username = None
                except StopIteration:
                    username = None

        message = ["/accept " + username]
        await self.send_message('', message)

    async def search_for_match(self, battle_format, team):
        await self.update_team(team)
        message = ["/search {}".format(battle_format)]
        await self.send_message("", message)

    async def leave_battle(self, battle_tag, save_replay=False):
        if save_replay:
            await self.save_replay(battle_tag)

        message = ["/leave {}".format(battle_tag)]
        await self.send_message('', message)

        while True:
            msg = await self.receive_message()
            if battle_tag in msg and 'deinit' in msg:
                return

    # TODO: delete this function since I don't think it ever gets used
    async def save_replay(self, battle_tag):
        message = ["/savereplay"]
        await self.send_message(battle_tag, message)

        while True:
            msg = await self.receive_message()
            if msg.startswith("|queryresponse|savereplay|"):
                obj = json.loads(msg.replace("|queryresponse|savereplay|", ""))
                log = obj['log']
                identifier = obj['id']
                post_response = requests.post(
                    "https://play.pokemonshowdown.com/~~showdown/action.php?act=uploadreplay",
                    data={
                        "log": log,
                        "id": identifier
                    }
                )
                if post_response.status_code != 200:
                    raise SaveReplayError("POST to save replay did not return a 200: {}".format(post_response.content))
                break
