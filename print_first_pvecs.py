# This utility file loads two saves, starts a game, prints the pvecs for the first turn for both sides, and exits.
import pkmn

assert(pkmn.resume)
env = pkmn.pkmn_env()
env.seed(pkmn.env_seed)
observation = env.reset(pkmn.choose_starting_pokemon())
report_observation = pkmn.bookkeeper.construct_observation_handler()
x, opp_x =  report_observation(observation)
print(pkmn.choose_action(x, pkmn.bookkeeper, env.action_space))
print(pkmn.opponent_choose_action(opp_x, pkmn.bookkeeper, env.opponent_action_space))

