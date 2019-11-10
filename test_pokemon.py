import unittest
import numpy as np

import pkmn
import pickle
from preprocess_observation import preprocess_observation as preprocess_observation
from bookkeeper import Bookkeeper as Bookkeeper

class TestPokemon(unittest.TestCase):
    
    def test_sigmoid(self):
        assert pkmn.sigmoid(0) == 0.5
    def test_relu(self):
        # basic tests
        weights = np.array([[2,7,1,8], [6,0,2,2]])
        biases = np.array([1, 3]).T
        biases.shape = (biases.shape[0],1)
        x = np.array([3,1,4,5]).T
        x.shape = (x.shape[0],1)
        assert np.all(pkmn.relu_hidden_layer(weights, biases, x) == [[58],[39]])

        weights = np.array([[2,-4,0,0], [-1,3,3,-7]])
        biases = np.array([0, 0]).T
        biases.shape = (biases.shape[0],1)
        x = np.array([6,9,4,2]).T
        x.shape = (x.shape[0],1)
        assert np.all(pkmn.relu_hidden_layer(weights, biases, x) == [[0],[19]])

    def test_loaded_model(self):
        f = open('save_stable.p', 'rb')
        pkmn.model = pickle.load(f)
        f.close()
        bookkeeper = pkmn.Bookkeeper(pkmn.render, pkmn.model)
        env, observation = pkmn.construct_environment()
        report_observation = bookkeeper.construct_observation_handler()
        x = report_observation(observation)
        action = pkmn.choose_action(x, bookkeeper)
        # Judgement and shadow claw are garbage; so is switching
        assert(action == "move 1" or action == "move 3")
    
    def test_preprocessing(self):
        # First test: give it a plausible start state. Note that the format of the input to preprocess_observation
        # is not finalised. If I can get the simulator to return a JSON object of the state, we might be looking
        # at two preprocess_observation classes; one for fighting people online (where you can't just muck around
        # in the guts of the game and get it to return what you want) and one for training.
        state_changes, indices = preprocess_observation('{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Shadow Claw","id":"shadowclaw","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Aerial Ace","id":"aerialace","pp":32,"maxpp":32,"target":"any","disabled":false},{"move":"Judgment","id":"judgment","pp":16,"maxpp":16,"target":"normal","disabled":false}]}],"side":{"name":"Bob","id":"p2","pokemon":[{"ident":"p2: Arceus","details":"Arceus-Fighting","condition":"402/402","active":true,"stats":{"atk":245,"def":297,"spa":297,"spd":297,"spe":297},"moves":["thunderbolt","shadowclaw","aerialace","judgment"],"baseAbility":"multitype","item":"fistplate","pokeball":"pokeball","ability":"multitype"},{"ident":"p2: Aggron","details":"Aggron, L10, M","condition":"39/39","active":false,"stats":{"atk":32,"def":46,"spa":22,"spd":22,"spe":20},"moves":["roar","heavyslam","rockslide","toxic"],"baseAbility":"sturdy","item":"aggronite","pokeball":"pokeball","ability":"sturdy"},{"ident":"p2: Dragonite","details":"Dragonite, L10, F","condition":"43/43","active":false,"stats":{"atk":37,"def":29,"spa":30,"spd":30,"spe":26},"moves":["dragonclaw","firepunch","roost","earthquake"],"baseAbility":"multiscale","item":"lumberry","pokeball":"pokeball","ability":"multiscale"},{"ident":"p2: Uxie","details":"Uxie, L10","condition":"40/40","active":false,"stats":{"atk":25,"def":36,"spa":25,"spd":36,"spe":29},"moves":["psyshock","yawn","stealthrock","psychic"],"baseAbility":"levitate","item":"leftovers","pokeball":"pokeball","ability":"levitate"},{"ident":"p2: Cacturne","details":"Cacturne, L10, F","condition":"39/39","active":false,"stats":{"atk":33,"def":22,"spa":33,"spd":22,"spe":21},"moves":["swordsdance","seedbomb","suckerpunch","spikes"],"baseAbility":"waterabsorb","item":"leftovers","pokeball":"pokeball","ability":"waterabsorb"},{"ident":"p2: Druddigon","details":"Druddigon, L10, F","condition":"40/40","active":false,"stats":{"atk":34,"def":28,"spa":22,"spd":28,"spe":19},"moves":["dragontail","suckerpunch","gunkshot","aerialace"],"baseAbility":"sheerforce","item":"lifeorb","pokeball":"pokeball","ability":"sheerforce"}]},"State":["40/40","272/272","46/46","42/42","37/37","41/41",1]}\nDEADBEEF\n')
        assert(np.all(indices == [2, 1, 5,3,6,4])) # The order is Arceus/Aggron/Dragonite/Uxie/Cacturne/Druddigon.So Aggron is in position 2, Arceus is in position 1, Cacturne is in position 5, Dragonite is in position 3, Druddigon is in position 6, Uxie is in position 4.
        assert([0,1] in state_changes)  # Our opponent has Ledian, which is Pokemon 1, on the field
        assert([1,1] in state_changes)  # Our AI has Arceus on the field, which is coincidentally also Pokemon 1
        assert([3,272] in state_changes)  # Ledian, which is index 3 overall, has 272 health
        assert([9,402] in state_changes)  # Arceus, which is position 9 overall, has 402 health
        # Second test: Give it another plausible start state.
        state_changes, indices = preprocess_observation('{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Shadow Claw","id":"shadowclaw","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Aerial Ace","id":"aerialace","pp":32,"maxpp":32,"target":"any","disabled":false},{"move":"Judgment","id":"judgment","pp":16,"maxpp":16,"target":"normal","disabled":false}]}],"side":{"name":"Bob","id":"p2","pokemon":[{"ident":"p2: Arceus","details":"Arceus-Fighting","condition":"402/402","active":true,"stats":{"atk":245,"def":297,"spa":297,"spd":297,"spe":297},"moves":["thunderbolt","shadowclaw","aerialace","judgment"],"baseAbility":"multitype","item":"fistplate","pokeball":"pokeball","ability":"multitype"},{"ident":"p2: Aggron","details":"Aggron, L10, M","condition":"39/39","active":false,"stats":{"atk":32,"def":46,"spa":22,"spd":22,"spe":20},"moves":["roar","heavyslam","rockslide","toxic"],"baseAbility":"sturdy","item":"aggronite","pokeball":"pokeball","ability":"sturdy"},{"ident":"p2: Dragonite","details":"Dragonite, L10, F","condition":"43/43","active":false,"stats":{"atk":37,"def":29,"spa":30,"spd":30,"spe":26},"moves":["dragonclaw","firepunch","roost","earthquake"],"baseAbility":"multiscale","item":"lumberry","pokeball":"pokeball","ability":"multiscale"},{"ident":"p2: Uxie","details":"Uxie, L10","condition":"40/40","active":false,"stats":{"atk":25,"def":36,"spa":25,"spd":36,"spe":29},"moves":["psyshock","yawn","stealthrock","psychic"],"baseAbility":"levitate","item":"leftovers","pokeball":"pokeball","ability":"levitate"},{"ident":"p2: Cacturne","details":"Cacturne, L10, F","condition":"39/39","active":false,"stats":{"atk":33,"def":22,"spa":33,"spd":22,"spe":21},"moves":["swordsdance","seedbomb","suckerpunch","spikes"],"baseAbility":"waterabsorb","item":"leftovers","pokeball":"pokeball","ability":"waterabsorb"},{"ident":"p2: Druddigon","details":"Druddigon, L10, F","condition":"40/40","active":false,"stats":{"atk":34,"def":28,"spa":22,"spd":28,"spe":19},"moves":["dragontail","suckerpunch","gunkshot","aerialace"],"baseAbility":"sheerforce","item":"lifeorb","pokeball":"pokeball","ability":"sheerforce"}]},"State":["40/40","272/272","46/46","42/42","37/37","41/41",4]}\nDEADBEEF\n')
        assert(np.all(indices == [2, 1, 5,3,6,4]))
        assert([0,4] in state_changes)  # Now our opponent has Swellow on the field
        assert([1,1] in state_changes)
        assert([6,37] in state_changes)
        assert([9,402] in state_changes)

    def test_bookkeeper(self):
        f = open('save_stable.p', 'rb')
        pkmn.model = pickle.load(f)
        f.close()
        bookkeeper = Bookkeeper(pkmn.render, pkmn.model)
        report_observation = bookkeeper.construct_observation_handler()
        # test report_observation
        report_observation('{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Shadow Claw","id":"shadowclaw","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Aerial Ace","id":"aerialace","pp":32,"maxpp":32,"target":"any","disabled":false},{"move":"Judgment","id":"judgment","pp":16,"maxpp":16,"target":"normal","disabled":false}]}],"side":{"name":"Bob","id":"p2","pokemon":[{"ident":"p2: Arceus","details":"Arceus-Fighting","condition":"69/402","active":true,"stats":{"atk":245,"def":297,"spa":297,"spd":297,"spe":297},"moves":["thunderbolt","shadowclaw","aerialace","judgment"],"baseAbility":"multitype","item":"fistplate","pokeball":"pokeball","ability":"multitype"},{"ident":"p2: Aggron","details":"Aggron, L10, M","condition":"39/39","active":false,"stats":{"atk":32,"def":46,"spa":22,"spd":22,"spe":20},"moves":["roar","heavyslam","rockslide","toxic"],"baseAbility":"sturdy","item":"aggronite","pokeball":"pokeball","ability":"sturdy"},{"ident":"p2: Dragonite","details":"Dragonite, L10, F","condition":"43/43","active":false,"stats":{"atk":37,"def":29,"spa":30,"spd":30,"spe":26},"moves":["dragonclaw","firepunch","roost","earthquake"],"baseAbility":"multiscale","item":"lumberry","pokeball":"pokeball","ability":"multiscale"},{"ident":"p2: Uxie","details":"Uxie, L10","condition":"40/40","active":false,"stats":{"atk":25,"def":36,"spa":25,"spd":36,"spe":29},"moves":["psyshock","yawn","stealthrock","psychic"],"baseAbility":"levitate","item":"leftovers","pokeball":"pokeball","ability":"levitate"},{"ident":"p2: Cacturne","details":"Cacturne, L10, F","condition":"39/39","active":false,"stats":{"atk":33,"def":22,"spa":33,"spd":22,"spe":21},"moves":["swordsdance","seedbomb","suckerpunch","spikes"],"baseAbility":"waterabsorb","item":"leftovers","pokeball":"pokeball","ability":"waterabsorb"},{"ident":"p2: Druddigon","details":"Druddigon, L10, F","condition":"40/40","active":false,"stats":{"atk":34,"def":28,"spa":22,"spd":28,"spe":19},"moves":["dragontail","suckerpunch","gunkshot","aerialace"],"baseAbility":"sheerforce","item":"lifeorb","pokeball":"pokeball","ability":"sheerforce"}]},"State":["40/40","42/272","46/46","42/42","37/37","41/41",1]}\nDEADBEEF\n')
        assert(bookkeeper.state[0] == 1)
        assert(bookkeeper.state[1] == 1)
        assert(bookkeeper.state[3] == 42)
        assert(bookkeeper.state[9] == 69)
        # test report_reward
        cur_reward = bookkeeper.reward_sum
        bookkeeper.report_reward(42)
        assert(bookkeeper.reward_sum == cur_reward + 42)
        assert(np.all(bookkeeper.rewards == [42]))
        # test report
        x = np.array([-1, -1, 100, 100, 100, 100, 100, 100,   100, 100, 100, 100, 100, 100])
        x.shape = (14,1)
        f = open('save_stable.p', 'rb')
        pkmn.model = pickle.load(f)
        f.close()
        pvec, h = pkmn.policy_forward(x)
        action = 3
        # Report the same data three times, because why not?
        bookkeeper.report(x, h, pvec, action)
        bookkeeper.report(x, h, pvec, action)
        bookkeeper.report(x, h, pvec, action)
        xs = np.vstack(bookkeeper.xs)
        assert(xs.shape == (3, 14)) # the variables appears to have been transposed, which isn't wrong but is wonky enough that I put investigating it onto the todo list
        hs = np.vstack(bookkeeper.hs)
        assert(hs.shape == (3, pkmn.H))
        pvecs = np.vstack(bookkeeper.pvecs)
        assert(pvecs.shape == (3, pkmn.A))
        actions = np.vstack(bookkeeper.actions)
        assert(actions.shape == (3, 1))

if __name__ == "__main__":
	unittest.main()
