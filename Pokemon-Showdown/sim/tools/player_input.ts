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
                        //Not a very user-friendly interface. In particular, the prompt may come before the omniscient stream has said its
                        //piece. But we don't want the user to put in their prompt until after this happens. (So that they know,
                        //for instance, what the opponent did on the previous turn). But it will work for testing.
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
