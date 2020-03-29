/**
 * Battle Stream
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Supports interacting with a PS battle in Stream format.
 *
 * This format is VERY NOT FINALIZED, please do not use it directly yet.
 *
 * @license MIT
 */

import * as Streams from './../lib/streams';
import {Battle} from './battle';

/**
 * Like string.split(delimiter), but only recognizes the first `limit`
 * delimiters (default 1).
 *
 * `"1 2 3 4".split(" ", 2) => ["1", "2"]`
 *
 * `Chat.splitFirst("1 2 3 4", " ", 1) => ["1", "2 3 4"]`
 *
 * Returns an array of length exactly limit + 1.
 */
function splitFirst(str: string, delimiter: string, limit: number = 1) {
	const splitStr: string[] = [];
	while (splitStr.length < limit) {
		const delimiterIndex = str.indexOf(delimiter);
		if (delimiterIndex >= 0) {
			splitStr.push(str.slice(0, delimiterIndex));
			str = str.slice(delimiterIndex + delimiter.length);
		} else {
			splitStr.push(str);
			str = '';
		}
	}
	splitStr.push(str);
	return splitStr;
}

export class BattleStream extends Streams.ObjectReadWriteStream<string> {
	debug: boolean;
	keepAlive: boolean;
	battle: Battle | null;

	constructor(options: {debug?: boolean, keepAlive?: boolean} = {}) {
		super();
		this.debug = !!options.debug;
		this.keepAlive = !!options.keepAlive;
		this.battle = null;
	}

	_write(chunk: string) {
		try {
			this._writeLines(chunk);
		} catch (err) {
			this.pushError(err);
			return;
		}
		if (this.battle) this.battle.sendUpdates();
	}

	_writeLines(chunk: string) {
		for (const line of chunk.split('\n')) {
			if (line.charAt(0) === '>') {
				const [type, message] = splitFirst(line.slice(1), ' ');
				this._writeLine(type, message);
			}
		}
	}

	_writeLine(type: string, message: string) {
		switch (type) {
		case 'start':
			const options = JSON.parse(message);
			options.send = (t: string, data: any) => {
				if (Array.isArray(data)) data = data.join("\n");
				this.push(`${t}\n${data}`);
				if (t === 'end' && !this.keepAlive) this.push(null);
			};
			if (this.debug) options.debug = true;
			this.battle = new Battle(options);
			break;
		case 'player':
			const [slot, optionsText] = splitFirst(message, ' ');
			this.battle!.setPlayer(slot as SideID, JSON.parse(optionsText));
			break;
		case 'p1':
		case 'p2':
		case 'p3':
		case 'p4':
			if (message === 'undo') {
				this.battle!.undoChoice(type);
			} else {
				this.battle!.choose(type, message);
			}
			break;
		case 'forcewin':
		case 'forcetie':
			this.battle!.win(type === 'forcewin' ? message as SideID : null);
			break;
		case 'tiebreak':
			this.battle!.tiebreak();
			break;
		}
	}

	_end() {
		// this is in theory synchronous...
		this.push(null);
		this._destroy();
	}

	_destroy() {
		if (this.battle) this.battle.destroy();
	}
}

/**
 * Splits a BattleStream into omniscient, spectator, p1, p2, p3 and p4
 * streams, for ease of consumption.
 */
export function getPlayerStreams(stream: BattleStream, name_to_index: anyObject) {
	const streams = {
		omniscient: new Streams.ObjectReadWriteStream({
			write(data: string) {
				stream.write(data);
			},
			end() {
				return stream.end();
			},
		}),
		spectator: new Streams.ObjectReadStream({
			read() {},
		}),
		p1: new Streams.ObjectReadWriteStream({
			write(data: string) {
				stream.write(data.replace(/(^|\n)/g, `$1>p1 `));
			},
		}),
		p2: new Streams.ObjectReadWriteStream({
			write(data: string) {
				stream.write(data.replace(/(^|\n)/g, `$1>p2 `));
			},
		}),
		p3: new Streams.ObjectReadWriteStream({
			write(data: string) {
				stream.write(data.replace(/(^|\n)/g, `$1>p3 `));
			},
		}),
		p4: new Streams.ObjectReadWriteStream({
			write(data: string) {
				stream.write(data.replace(/(^|\n)/g, `$1>p4 `));
			},
		}),
	};
	(async () => {
		let chunk;
		// tslint:disable-next-line:no-conditional-assignment
		while ((chunk = await stream.read())) {
			const [type, data] = splitFirst(chunk, `\n`);
			switch (type) {
			case 'update':
				//This is the guts of the battle system. Whenever the battle says update, it pushes the information it needs
				// to the relevant AIs. For the most part, you want to use SideUpdate instead.
				streams.omniscient.push(Battle.extractUpdateForSide(data, 'omniscient'));
				streams.p1.push(Battle.extractUpdateForSide(data, 'p1'));
				streams.p2.push(Battle.extractUpdateForSide(data, 'p2'));
				break;
			case 'sideupdate':
				//The sideupdate is some variation on a forced switch, a move request, or a team preview.
				//It provides everything you could want to know about your side... But nothing about the opponent's side.
				//An AI which knows nothing about the opposing team is doomed to failure, so scrape that information.
				let supplementary_data = [];
				//Initialise the other_side_index because we need it in this scope.
				let other_side_index = 42;
				if (data[1] == 1) {
					other_side_index = 1;
				} else if (data[1] == 2) {
					other_side_index = 0;
				} else {
					console.log(data[1]);
					console.log(crash);
				}
                                /*let pokemonIndices = [name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[0].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[1].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[2].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[3].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[4].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[5].id]];*/
				// Over the long term, the above is how the pokemonIndices variable is going to get assigned. But, for now,
				// we have only four Pokemon on the field.
				let pokemonIndices = [name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[0].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[1].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[2].id],
				name_to_index[other_side_index][stream.battle.sides[other_side_index].pokemon[3].id],
				4,5];
				let our_side_index = data[1] - 1;
				let ourPokemonIndices = [name_to_index[our_side_index][stream.battle.sides[our_side_index].pokemon[0].id],
				name_to_index[our_side_index][stream.battle.sides[our_side_index].pokemon[1].id],
				name_to_index[our_side_index][stream.battle.sides[our_side_index].pokemon[2].id],
				name_to_index[our_side_index][stream.battle.sides[our_side_index].pokemon[3].id],
				4,5];
				supplementary_data = ['0 fnt','0 fnt','0 fnt','0 fnt','0 fnt','0 fnt',  '0 fnt','0 fnt','0 fnt','0 fnt','0 fnt','0 fnt',  '0',  '0','0','0','0','0','0', '0',  '0','0','0','0','0', '0', '0',  '', '', '', '', '','','','','','','','','','','',''];
				supplementary_data[12] = pokemonIndices[0];
				// Similarly, this will emerge from its commented-out glory in the near future.
				//for (let i in [0,1,2,3,4,5]) {
				for (let i in [0,1,2,3]) {
					// There is probably a way to get this information with fewer operations.
					supplementary_data[ourPokemonIndices[i]] = stream.battle.sides[our_side_index].pokemon[i].getDetails().shared.split("|")[1];
					if (stream.battle.sides[our_side_index].pokemon[i].status) {
						// If the Pokemon is asleep or badly poisoned, add how many turns it has left/how many turns it has been active.
						// THIS COULD BE CLEANED UP
						if (stream.battle.sides[our_side_index].pokemon[i].statusData.time) {
							supplementary_data[ourPokemonIndices[i]] += " 1";
						}
						if (stream.battle.sides[our_side_index].pokemon[i].statusData.stage) {
							supplementary_data[ourPokemonIndices[i]] += " 1";
						}
					}
				}

				for (let i in [0,1,2,3]) {
					supplementary_data[pokemonIndices[i] + 6] = stream.battle.sides[other_side_index].pokemon[i].getDetails().shared.split("|")[1];
					if (stream.battle.sides[other_side_index].pokemon[i].status) {
						if (stream.battle.sides[other_side_index].pokemon[i].status != "fnt") {
							if (stream.battle.sides[other_side_index].pokemon[i].statusData.time) {
								supplementary_data[pokemonIndices[i] + 6] += " " + JSON.stringify(stream.battle.sides[other_side_index].pokemon[i].statusData.time);
							}
							if (stream.battle.sides[other_side_index].pokemon[i].statusData.stage) {
								supplementary_data[pokemonIndices[i] + 6] += " " + JSON.stringify(stream.battle.sides[other_side_index].pokemon[i].statusData.stage);
							}
						}
					}
				}
				// Add data about both side's stat boosts to supplementary_data.
				let osi_ind = 13;
				for (let i in stream.battle.sides[our_side_index].pokemon[0].boosts) {
					supplementary_data[osi_ind] = stream.battle.sides[our_side_index].pokemon[0].boosts[i];
					osi_ind += 1;
				}
				for (let i in stream.battle.sides[other_side_index].pokemon[0].boosts) {
					supplementary_data[osi_ind] = stream.battle.sides[other_side_index].pokemon[0].boosts[i];
					osi_ind += 1;
				}
				// Add other information.
				supplementary_data[27] = stream.battle.field.weather;
				supplementary_data[28] = stream.battle.field.terrain;
				supplementary_data[29] = Object.keys(stream.battle.sides[our_side_index].sideConditions);
				supplementary_data[30] = Object.keys(stream.battle.sides[other_side_index].sideConditions);
				// Add the item of each Pokemon (which gets preprocessed into whether it has an item).
				for (let i of [0,1,2,3]) {
					supplementary_data[31+ourPokemonIndices[i]] = stream.battle.sides[our_side_index].pokemon[i].item;
					supplementary_data[37+pokemonIndices[i]] = stream.battle.sides[other_side_index].pokemon[i].item;
				}
				//Stitch it together.
				supplementary_data = ',"State":' + JSON.stringify(supplementary_data) + "}"
				const [side, sideData] = splitFirst(data.slice(0, -1) + supplementary_data, `\n`);
				streams[side as SideID].push(sideData);
				break;
			case 'end':
				let result = JSON.parse(data);
				if (result.winner == "Alice") {
					streams.p1.push('|request|{"Victory":"yes"}'); 
					streams.p2.push('|request|{"Victory":"no"}'); 
				} else {
					if (result.winner == "HughMann") {
						streams.p1.push('|request|{"Victory":"no"}'); 
						streams.p2.push('|request|{"Victory":"yes"}'); 
					} else {
						//The winner was someone other than Alice or HughMann. This can't be good.
						console.log(crash);
					}
				}
				break;
			}
		}
		for (const s of Object.values(streams)) {
			s.push(null);
		}
	})().catch(err => {
		for (const s of Object.values(streams)) {
			s.pushError(err);
		}
	});
	return streams;
}

export abstract class BattlePlayer {
	readonly stream: Streams.ObjectReadWriteStream<string>;
	readonly log: string[];
	readonly debug: boolean;

	constructor(playerStream: Streams.ObjectReadWriteStream<string>, debug: boolean = false) {
		this.stream = playerStream;
		this.log = [];
		this.debug = debug;
	}

	async start() {
		let chunk;
		// tslint:disable-next-line:no-conditional-assignment
		while ((chunk = await this.stream.read())) {
			this.receive(chunk);
		}
	}

	receive(chunk: string) {
		for (const line of chunk.split('\n')) {
			this.receiveLine(line);
		}
	}

	receiveLine(line: string) {
		if (line.charAt(0) !== '|') return;
		const [cmd, rest] = splitFirst(line.slice(1), '|');
		if (cmd === 'request') return this.receiveRequest(JSON.parse(rest));
		if (cmd === 'error') return this.receiveError(new Error(rest));
		this.log.push(line);
	}

	abstract receiveRequest(request: AnyObject): void;

	receiveError(error: Error) {
		throw error;
	}
	
	//This is a method used by the classes which inherit from this one. 
	getState(request: AnyObject) {
		//This method is not supposed to be used, so crash if it gets called.
		console.log(crash);
	}

	choose(choice: string) {
		this.stream.write(choice);
	}
}

export class BattleTextStream extends Streams.ReadWriteStream {
	readonly battleStream: BattleStream;
	currentMessage: string;

	constructor(options: {debug?: boolean}) {
		super();
		this.battleStream = new BattleStream(options);
		this.currentMessage = '';
	}

	async start() {
		let message;
		// tslint:disable-next-line:no-conditional-assignment
		while ((message = await this.battleStream.read())) {
			if (!message.endsWith('\n')) message += '\n';
			this.push(message + '\n');
		}
		this.push(null);
	}

	_write(message: string | Buffer) {
		this.currentMessage += '' + message;
		const index = this.currentMessage.lastIndexOf('\n');
		if (index >= 0) {
			this.battleStream.write(this.currentMessage.slice(0, index));
			this.currentMessage = this.currentMessage.slice(index + 1);
		}
	}

	_end() {
		return this.battleStream.end();
	}
}
