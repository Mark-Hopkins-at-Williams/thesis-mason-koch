# Setup
Change directory to Pokemon-Showdown. Run "node build". (If you haven't installed node yet, you will need to). Then cd .. back to the previous directory. If you want to load example weights, rename save\_stable.p to save.p. If you want to start training from scratch, set the resume variable in pkmn.py to "False". To start learning, run "python pkmn.py". Note that env_pkmn.py, which pkmn.py depends on, uses a relative path to get to its dependences in Pokemon-Showdown. So if you run this from some other directory, it will be broken.

env_pkmn.py runs test\_random\_player from the sim/examples directory. This is based on the battle-stream-example from the same directory. You can also run this from inside the Pokemon-Showdown directory, and it will let you fight the random AI from the command line. It in turn uses random-player-ai from the sim/tools directory. This is provided by Pokemon Showdown. It also uses player\_input from the same directory. This is a stripped down version of random-player-ai which will prompt the user for an input from the command line.

test\_random\_player also depends on battle_stream from the sim directory. The while loop starting at line 160 has been edited so that it writes information about the other side in addition to information about one's own side.




