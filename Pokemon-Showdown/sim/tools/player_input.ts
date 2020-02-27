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
	                        console.log('actionspace' + JSON.stringify([[]]));
	                        console.log("gameinfo" + JSON.stringify(request));
	                        console.log("DEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify([[]]));
	                        console.log("DEADBEEF");
				if (this.name != "Alice") {
					console.log(crash);
				}
			}

		} else if (request.forceSwitch) {
			// Check switches for legality. Adapted from random-player-ai
                        const pokemon = request.side.pokemon;
                        const chosen: number[] = [];
                        const choices = request.forceSwitch.map((mustSwitch: AnyObject) => {
                                if (!mustSwitch) return `pass`;

                                const canSwitch = [1, 2, 3, 4, 5, 6].filter(i => (
                                        pokemon[i - 1] &&
                                        // not active
                                        i > request.forceSwitch.length &&
                                        // not chosen for a simultaneous switch
                                        !chosen.includes(i) &&
                                        // not fainted
                                        !pokemon[i - 1].condition.endsWith(` fnt`)
                                ));

                                if (!canSwitch.length) return `pass`;

                                let retswitches = [];
                                for (let i of canSwitch) {
                                    retswitches.push('switch ' + pokemon[i-1].ident.substring(4).toLowerCase());
                                }
                                return [[], retswitches];
                        });
                        // 2ms timeout. This is a relic from when the input was a stream and likely doesn't do anything now.
                        // It is still here because it does affect runtime in a statistically significant manner.
                        await new Promise(resolve => setTimeout(resolve, 2));;
			if (this.name === "HughMann") {
	                        console.log('actionspace' + JSON.stringify(choices));
	                        console.log("gameinfo" + JSON.stringify(request));
	                        // This lets pkmn_env.py stop reading input. This is something I should have done a long time ago.
	                        console.log("DEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify(choices));
	                        console.log("DEADBEEF");
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
			//this.choose(ans);
		} else if (request.active) {
                        // Check switches and moves for legality
                        let [canMegaEvo, canUltraBurst, canZMove] = [true, true, true];
                        const pokemon = request.side.pokemon;
                        const chosen: number[] = [];
                        const choices = request.active.map((active: AnyObject, i: number) => {
                                if (pokemon[i].condition.endsWith(` fnt`)) return `pass`;
                                canMegaEvo = canMegaEvo && active.canMegaEvo;
                                canUltraBurst = canUltraBurst && active.canUltraBurst;
                                canZMove = canZMove && !!active.canZMove;
                                let canMove = [1, 2, 3, 4].slice(0, active.moves.length).filter(j => (
                                        // not disabled
                                        !active.moves[j - 1].disabled
                                        // NOTE: we don't actually check for whether we have PP or not because the
                                        // simulator will mark the move as disabled if there is zero PP and there are
                                        // situations where we actually need to use a move with 0 PP (Gen 1 Wrap).
                                )).map(j => ({
                                        slot: j,
                                        move: active.moves[j - 1].move,
                                        target: active.moves[j  - 1].target,
                                        zMove: false,
                                }));
                                if (canZMove) {
                                        canMove.push(...[1, 2, 3, 4].slice(0, active.canZMove.length)
                                                .filter(j => active.canZMove[j - 1])
                                                .map(j => ({
                                                        slot: j,
                                                        move: active.canZMove[j - 1].move,
                                                        target: active.canZMove[j - 1].target,
                                                        zMove: true,
                                                })));
                                }
                                // Filter out adjacentAlly moves if we have no allies left, unless they're our
                                // only possible move options.
                                // Unlike other changes in this directory, this one might be permanent. The chance that we will ever care about doubles is very low.
                                const hasAlly = false//!pokemon[i ^ 1].condition.endsWith(` fnt`);
                                const filtered = canMove.filter(m => m.target !== `adjacentAlly` || hasAlly);
                                canMove = filtered.length ? filtered : canMove;
                                const moves = canMove.map(m => {
                                        let move = `move ${m.slot}`;
                                        // NOTE: We don't generate all possible targeting combinations.
                                        if (request.active.length > 1) {
                                                if ([`normal`, `any`, `adjacentFoe`].includes(m.target)) {
                                                        move += ` ${1 + Math.floor(this.prng.next() * 2)}`;
                                                }
                                                if (m.target === `adjacentAlly`) {
                                                        move += ` -${(i ^ 1) + 1}`;
                                                }
                                                if (m.target === `adjacentAllyOrSelf`) {
                                                        if (hasAlly) {
                                                                move += ` -${1 + Math.floor(this.prng.next() * 2)}`;
                                                        } else {
                                                                move += ` -${i + 1}`;
                                                        }
                                                }
                                        }
                                        if (m.zMove) move += ` zmove`;
                                        return {choice: move, move: m};
                                });

                                const canSwitch = [1, 2, 3, 4, 5, 6].filter(j => (
                                        pokemon[j - 1] &&
                                        // not active
                                        !pokemon[j - 1].active &&
                                        // not chosen for a simultaneous switch
                                        !chosen.includes(j) &&
                                        // not fainted
                                        !pokemon[j - 1].condition.endsWith(` fnt`)
                                ));
                                const switches = active.trapped ? [] : canSwitch;
                                let retswitches = [];
                                for (let i of switches) {
                                    retswitches.push('switch ' + pokemon[i-1].ident.substring(4).toLowerCase());
                                }
                                return [moves, retswitches];
                        });
                        await new Promise(resolve => setTimeout(resolve, 2));;

			if (this.name === "HughMann") {
	                        console.log('actionspace' + JSON.stringify(choices));
	                        console.log("gameinfo" + JSON.stringify(request));
	                        console.log("DEADBEEF");
			} else {
				console.log('opponentspace' + JSON.stringify(choices));
	                        console.log("DEADBEEF");
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
			//this.choose(ans);
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
