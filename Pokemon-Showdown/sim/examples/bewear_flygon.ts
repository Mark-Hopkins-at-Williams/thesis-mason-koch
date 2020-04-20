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
	team: 'Flygon||lumberry|levitate|earthquake,outrage,firepunch,dragondance||85,85,85,85,85,85||||82|]Tapu Koko||choiceband|electricsurge|closecombat,bravebird,playrough,wildcharge||85,85,85,85,85,85|N|||80|]Swoobat||lifeorb|simple|heatwave,psyshock,nastyplot,airslash||85,,85,85,85,85||,0,,,,||88|]Lunala||leftovers|shadowshield|calmmind,roost,psyshock,moongeistbeam||85,,85,85,85,85|N|,0,,,,||72|]Persian||choiceband|limber|uturn,knockoff,doubleedge,playrough||85,85,85,85,85,85||||88|]Jolteon||choicespecs|voltabsorb|shadowball,hypervoice,thunderbolt,voltswitch||85,,85,85,85,85||,0,,,,||86|',
};
const p2spec = {
	name: "HughMann",
	team: 'Bewear||lifeorb|fluffy|swordsdance,doubleedge,darkestlariat,closecombat||85,85,85,85,85,85||||84|]Cofagrigus||leftovers|mummy|memento,bodypress,shadowball,willowisp||85,85,85,85,85,85||||86|]Stonjourner||lifeorb|powerspot|rockpolish,earthquake,stoneedge,heatcrash||85,85,85,85,85,85||||88|]Gothitelle||leftovers|competitive|nastyplot,thunderbolt,psychic,shadowball||85,,85,85,85,85||,0,,,,||88|]Rhyperior||weaknesspolicy|solidrock|earthquake,rockpolish,megahorn,stoneedge||85,85,85,85,85,85||||82|]Slowbro||leftovers|regenerator|calmmind,psyshock,scald,icebeam||85,,85,85,85,85||,0,,,,||82|',
};

const p1lookup = {
	name: "Alice",
	team: 'Flygon||lumberry|levitate|earthquake,outrage,firepunch,dragondance||85,85,85,85,85,85||||82|]Tapu Koko||choiceband|electricsurge|closecombat,bravebird,playrough,wildcharge||85,85,85,85,85,85|N|||80|]Swoobat||lifeorb|simple|heatwave,psyshock,nastyplot,airslash||85,,85,85,85,85||,0,,,,||88|]Lunala||leftovers|shadowshield|calmmind,roost,psyshock,moongeistbeam||85,,85,85,85,85|N|,0,,,,||72|]Persian||choiceband|limber|uturn,knockoff,doubleedge,playrough||85,85,85,85,85,85||||88|]Jolteon||choicespecs|voltabsorb|shadowball,hypervoice,thunderbolt,voltswitch||85,,85,85,85,85||,0,,,,||86|',
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

const aliceLookup = spec_to_dict(p1lookup, "p1");
const bobLookup = spec_to_dict(p2spec, "p2");

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

