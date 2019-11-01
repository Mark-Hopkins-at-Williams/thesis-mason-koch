/**
 * Takes input from the user to run a battle.
 */

import {ObjectReadWriteStream} from '../../lib/streams';
import {BattlePlayer} from '../battle-stream';
import {PRNG, PRNGSeed} from '../prng';

export class Player_input extends BattlePlayer {
	protected readonly move: number;
	protected readonly mega: number;
	protected readonly prng: PRNG;

	constructor(
		playerStream: ObjectReadWriteStream<string>,
		options: {move?: number, mega?: number, seed?: PRNG | PRNGSeed | null } = {},
		debug: boolean = false,
		name: string = "NoneGiven"
	) {
		super(playerStream, debug);
		this.move = options.move || 1.0;
		this.mega = options.mega || 0;
		this.prng = options.seed && !Array.isArray(options.seed) ? options.seed : new PRNG(options.seed);
		//The player name is used for debugging and writing weights to file.
		this.name = name;
		this.reader = require("readline");
	}

        //https://stackoverflow.com/questions/18193953/waiting-for-user-to-enter-input-in-node-js
        askQuestion(query) {
                const rl = this.reader.createInterface({
                        input: process.stdin,
                        output: process.stdout,
                });
                return new Promise(resolve => rl.question(query, ans => {
                        rl.close();
                        resolve(ans);
                }));
        }

	receiveError(error: Error) {
		console.log(error);
		console.log(crash);
	}

	newStream(playerStream: ObjectReadWriteStream<string>) {
		this.stream = playerStream;
	}

	async receiveRequest(request: AnyObject) {
		if (request.wait) {
			// wait request. do nothing.
		} else if (request.forceSwitch || request.active) {
                        //Update: We need to wait for the omniscient stream to say its piece. This code should be changed soon.
                        await new Promise(resolve => setTimeout(resolve, 10));;
                        // In the future, it should be possible to copy-paste the code from I think it was random_player_AI to see which actions are valid.
                        // This will be more consistent than the current solution.
                        for (let pkmn in [0,1,2,3,4,5]){
                            console.log(request.side.pokemon[pkmn].ident);
                        }
                        // This lets pkmn_env.py to stop reading input. This is something I should have done a long time ago.
                        console.log("DEADBEEF");
                        let ans = await this.askQuestion("Provide input: ");
                        if (ans == "help") {
                                console.log(request.side.pokemon);
                                ans = await this.askQuestion("There you go, now provide input: ");
                        }
                        this.choose(ans);
                } else if (request.Victory == "yes" || request.Victory == "no") {
                        console.log(this.name + request.Victory);
                } else {
			// team preview?
			this.choose(this.chooseTeamPreview(request.side.pokemon));
		}
	}

	protected chooseTeamPreview(team: AnyObject[]): string {
		return `default`;
	}
}
