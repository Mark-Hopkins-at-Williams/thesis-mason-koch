/**
 * Tests random AI against human.
 */
import {BattleStream, getPlayerStreams} from '../battle-stream';
import {Dex} from '../dex';
import {MyAI} from '../tools/my_ai';
import {RandomPlayerAI} from '../tools/random-player-ai';
import {Player_input} from '../tools/player_input';
// tslint:disable:no-floating-promises
const spec = {
	formatid: "gen8customgame",
	seed: process.argv[2],
};

if (process.argv[2]) {
	spec.seed = JSON.parse(process.argv[2]);
}


const p1spec = {
	name: "Alice",
	team: 'Dragapult||spelltag|infiltrator|hex,will-o-wisp,dracometeor,u-turn|timid|,,,252,4,252||||100|]Toxapex||blacksludge|regenerator|scald,recover,haze,toxic|bold|252,,252,,4,||||100|]Dugtrio||choicescarf|arenatrap|earthquake,stoneedge,shadowclaw,memento|adamant|,252,4,,,252||||100|]Clefable||leftovers|magicguard|teleport,moonblast,wish,protect|calm|252,,4,,252,||||100|]Mandibuzz||heavy-dutyboots|overcoat|foulplay,roost,u-turn,defog|careful|252,,,,160,96||||100|]Ferrothorn||leftovers|ironbarbs|spikes,bodypress,knockoff,leechseed|impish|252,,252,,4,||||100|',
};
const p2spec = {
	name: "HughMann",
	team: 'Corviknight||leftovers|pressure|ironhead,defog,roost,bodypress|careful|252,,48,,208,||||100|]Dugtrio||choicescarf|arenatrap|earthquake,stoneedge,shadowclaw,memento|adamant|,252,4,,,252||||100|]Mandibuzz||heavy-dutyboots|overcoat|foulplay,roost,u-turn,defog|careful|252,,,,160,96||||100|]Rotom-heat||heavy-dutyboots|levitate|nastyplot,overheat,voltswitch,toxic|timid|248,,,8,,252||||100|]Seismitoad||leftovers|waterabsorb|stealthrock,toxic,earthpower,scald|bold|252,,252,,4,||||100|]Clefable||leftovers|magicguard|teleport,moonblast,wish,protect|calm|252,,4,,252,||||100|',
};

const p2lookup = {
	name: "HughMann",
	team: 'Corviknight||leftovers|pressure|ironhead,defog,roost,bodypress|careful|252,,48,,208,||||100|]Dugtrio||choicescarf|arenatrap|earthquake,stoneedge,shadowclaw,memento|adamant|,252,4,,,252||||100|]Mandibuzz||heavy-dutyboots|overcoat|foulplay,roost,u-turn,defog|careful|252,,,,160,96||||100|]Rotom||heavy-dutyboots|levitate|nastyplot,overheat,voltswitch,toxic|timid|248,,,8,,252||||100|]Seismitoad||leftovers|waterabsorb|stealthrock,toxic,earthpower,scald|bold|252,,252,,4,||||100|]Clefable||leftovers|magicguard|teleport,moonblast,wish,protect|calm|252,,4,,252,||||100|',
};

function spec_to_dict(spec: anyObject, name: string) {
	let pokemonNames = spec.team.split('|');
	pokemonNames = [name + ": " + pokemonNames[0], name + ": " + pokemonNames[11].slice(1), name + ": " + pokemonNames[22].slice(1), name + ": " + pokemonNames[33].slice(1), name + ": " + pokemonNames[44].slice(1), name + ": " + pokemonNames[55].slice(1)];
	let pokemonIndices = [0,0,0,0,0,0]
	let retval = {};
	for (let i in [0,1,2,3,4,5]) {
		for (let j in [0,1,2,3,4,5]) {
			if (pokemonNames[j] < pokemonNames[i]) {
				pokemonIndices[i] += 1;
			}
		}
		retval[pokemonNames[i]] = pokemonIndices[i];
	}
	return retval;
}

const aliceLookup = spec_to_dict(p1spec, "p1");
const bobLookup = spec_to_dict(p2lookup, "p2");

//Set up the streaming infrastructure.
let streams = getPlayerStreams(new BattleStream(), [aliceLookup, bobLookup]);

//Debug is false.
const p1 = new Player_input(streams.p1, {}, false, "Alice");
const p2 = new Player_input(streams.p2, {}, false, "HughMann");

//Tell the players to start up.
void p1.start();
void p2.start();

/*void (async () => {
        let chunk;
        // tslint:disable-next-line no-conditional-assignment
        while ((chunk = await streams.omniscient.read())) {
                console.log(chunk);
        }
})();*/

void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

