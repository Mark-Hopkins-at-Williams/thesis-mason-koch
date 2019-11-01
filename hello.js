class ExampleCommunicator {
	constructor() {
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
	async run() {
		let ans = ""
		while (ans != "quit") {
			ans = await this.askQuestion("");
			console.log(ans + " is what you said!")
		}
	}
}

test = new ExampleCommunicator()
test.run()

