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
			if (this.name === "HughMann") {
				console.log('actionspace' + JSON.stringify([]) + "\ngameinfo" + JSON.stringify(request) + "\nDEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify([]) + "\nDEADBEEF");
				if (this.name != "Alice") {
					console.log(crash);
				}
			}

		} else if (request.forceSwitch) {

			// Check switches for legality. Adapted from random-player-ai
			const pokemon = request.side.pokemon;
			const canSwitch = [1, 2, 3, 4, 5, 6].filter(j => (
				pokemon[j - 1] &&
				// not active
				!pokemon[j - 1].active &&
				// not fainted
				!pokemon[j - 1].condition.endsWith(` fnt`)
			));
			let choices = [];
			for (let i of canSwitch) {
			    choices.push('switch ' + pokemon[i-1].ident.substring(4).toLowerCase());
			}
			if (this.name === "HughMann") {
				console.log('actionspace' + JSON.stringify(choices) + "\ngameinfo" + JSON.stringify(request) + "\nDEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify(choices)+"\nDEADBEEF");
				if (this.name != "Alice") {
					console.log(crash);
				}
			}
			let ans = await this.askQuestion("");
			var ans2 = ans.split('|')
			if (this.name == "Alice") {
				this.choose(ans2[0]);
			} else {
				this.choose(ans2[1]);
			}
		} else if (request.active) {
			// Check switches and moves for legality
			const pokemon = request.side.pokemon;
			const canSwitch = [1, 2, 3, 4, 5, 6].filter(j => (
				pokemon[j - 1] &&
				// not active
				!pokemon[j - 1].active &&
				// not fainted
				!pokemon[j - 1].condition.endsWith(` fnt`)
			));
			let choices = [];
			if (!request.active.trapped) {
				for (let i of canSwitch) {
				    choices.push('switch ' + pokemon[i-1].ident.substring(4).toLowerCase());
				}
			}
			for (let i of [0,1,2,3]) {
				if (!request.active[0]['moves'][i].disabled) {
					choices.push('move ' + (i+1));
				}
			}
			if (this.name === "HughMann") {
				console.log('actionspace' + JSON.stringify(choices) + "\ngameinfo" + JSON.stringify(request) + "\nDEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify(choices)+"\nDEADBEEF");
				if (this.name != "Alice") {
					console.log(crash);
				}
			}
			let ans = await this.askQuestion("");
			var ans2 = ans.split('|')
			if (this.name == "Alice") {
				this.choose(ans2[0]);
			} else {
				this.choose(ans2[1]);
			}
		} else if (request.Victory == "yes" || request.Victory == "no") {
			if (this.name === "HughMann") {
				console.log(this.name + request.Victory);
			} else {
				if (this.name != "Alice") {
					console.log(crash);
				}
			}
		} else {
			// team preview?
			let ans = await this.askQuestion("");
			var ans2 = ans.split('|')
			if (this.name == "Alice") {
				this.choose(ans2[0]);
			} else {
				this.choose(ans2[1]);
			}
		}
	}

	protected chooseTeamPreview(team: AnyObject[]): string {
		return `default`;
	}
}
