let creatures = [];
let population = 50;
let survivors = 1;
let genCount = 0;
let plate;
let highscore = 0;
let width = 512;
let height = 512;
let highestLifetime = 0;
let currentLifetime = 0
// let fps = 30;

function setup() {
    createCanvas(width, height);
    background(0);
    newGeneration();
    // frameRate(fps);
    plate = new Plate(width / 2, height / 2, 200);
}

function draw() {
    background(0);
    plate.draw();
    creatures.forEach(ball => {
        ball.think();
        ball.move();
        ball.checkDanger();
        ball.draw();
    });
    noStroke();
    drawGenCount();
    if (survivalOfTheFittest()) {
        currentLifetime = 0;
        createNewPopulation()
    }
}

function createNewPopulation() {
    calculateScore();
    let fitest = getFitest();
    creatures = [];
    for (let i = 1; i < population; i++) {
        const creature = fitest[i];
        let child;
        if (random() > 0.5) {
            child = fitest[0].copulate(creature);
        } else {
            child = creature.copulate(fitest[0]);
        }
        child.mutate();
        creatures.push(child);
    }
    fitest[0].reset();
    creatures.push(fitest[0]);
    genCount++;
}

function drawGenCount() {
    textSize(20);
    fill(255);
    textAlign(LEFT);
    text(`Generation: ${genCount}`, 10, 30);
    textSize(15);
    fill(255);
    textAlign(LEFT);
    text(`current lt: ${currentLifetime}`, 10, 80);
    if (currentLifetime > highestLifetime) {
        highestLifetime = currentLifetime;
    }
    currentLifetime++;
    text(`highest lt: ${highestLifetime}`, 10, 100);
}

function newGeneration() {
    creatures = [];
    for (let i = 0; i < population; i++) {
        creatures.push(new Creature(width / 2, height / 2 - 150, color(255, 175, 150, 100)));
    }
}

function getFitest() {
    let sorted = creatures.sort(function (a, b) {
        return (a.score - b.score) * -1;
    });
    return sorted;
};

function calculateScore() {
    for (let i = 0; i < creatures.length; i++) {
        const creature = creatures[i];
        creature.score += creature.livingtime;
    }
}

function survivalOfTheFittest() {
    let alive = 0
    for (let i = 0; i < creatures.length; i++) {
        const ball = creatures[i];
        if (ball.isAlive) alive++;
    }
    textSize(15);
    fill(255);
    textAlign(LEFT);
    text(`alive: ${alive}`, 10, 60);
    return alive < 1;
}

// This class is inspired by Siraj Reval(https://www.youtube.com/watch?v=HT1_BHA3ecY)
class NeuralNetwork {

    constructor(inputNodes, hiddenNodes, outputNodes) {

        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.inputWeights = tf.randomNormal([this.inputNodes, this.hiddenNodes]);
        this.outputWeights = tf.randomNormal([this.hiddenNodes, this.outputNodes]);
    }

    predict(input) {
        let output;
        tf.tidy(() => {
            let input_layer = tf.tensor(input, [1, this.inputNodes]);
            let hidden_layer = input_layer.matMul(this.inputWeights).sigmoid();
            let output_layer = hidden_layer.matMul(this.outputWeights).sigmoid();
            output = output_layer.dataSync();
        });
        return output;
    }

    clone() {
        let clonie = new NeuralNetwork(this.inputNodes, this.hiddenNodes, this.outputNodes);
        clonie.dispose();
        clonie.inputWeights = tf.clone(this.inputWeights);
        clonie.outputWeights = tf.clone(this.outputWeights);
        return clonie;
    }

    dispose() {
        this.inputWeights.dispose();
        this.outputWeights.dispose();
    }
}

class Creature {
    constructor(x, y, col) {
        this.score = 0;
        this.isAlive = true;
        this.pos = createVector(x, y);
        this.dir = createVector(1, 0);
        this.brain = new NeuralNetwork(3, 6, 2);
        this.color = col;
        this.radius = 20;
        this.livingtime = 0;
    }

    checkDanger() {
        let distance = p5.Vector.dist(this.pos, plate.pos)
        if (distance > plate.radius || distance < plate.radius / 2) {
            this.isAlive = false;
        }
    }

    think() {
        let memo = this.brain.predict([p5.Vector.dist(this.pos, plate.pos) / 200, this.dir.x, this.dir.y]);
        if (memo[0] >= 0.5) {
            this.moveleft(0.06);
            this.left = true;
        } else {
            this.left = false;
        }
        if (memo[1] >= 0.5) {
            this.moveright(0.06);
            this.right = true;
        } else {
            this.right = false;
        }
    }

    reset() {
        this.score = 0;
        this.isAlive = true;
        this.pos = createVector(width / 2, height / 2 - 150);
        this.dir = createVector(1, 0);
        this.color = color(50, 50, 200, 200);
        this.radius = 20;
        this.livingtime = 0;
    }

    move() {
        this.pos.add(this.dir.copy().mult(5.25));
    }

    moveleft(a) {
        this.dir.rotate(-a);
    }

    moveright(a) {
        this.dir.rotate(a);
    }

    copulate(partner) {
        let parentA_in_dna = this.brain.inputWeights.dataSync();
        let parentA_out_dna = this.brain.outputWeights.dataSync();
        let parentB_in_dna = partner.brain.inputWeights.dataSync();
        let parentB_out_dna = partner.brain.outputWeights.dataSync();

        let mid = Math.floor(random() * parentA_in_dna.length);
        // let mid = Math.floor(parentA_in_dna.length / 3);
        let child_in_dna = [...parentA_in_dna.slice(0, mid), ...parentB_in_dna.slice(mid, parentB_in_dna.length)];
        let child_out_dna = [...parentA_out_dna.slice(0, mid), ...parentB_out_dna.slice(mid, parentB_out_dna.length)];

        let input_shape = this.brain.inputWeights.shape;
        let output_shape = this.brain.outputWeights.shape;

        let child = new Creature(width / 2, height / 2 - 150, color(255, 175, 150, 100));
        // child.brain.dispose();

        child.brain.inputWeights = tf.tensor(child_in_dna, input_shape);
        child.brain.outputWeights = tf.tensor(child_out_dna, output_shape);

        return child;
    }

    mutate = () => {
        let mutateRate = 0.1;
        let inputWeights = this.brain.inputWeights.dataSync();
        let outputWeights = this.brain.outputWeights.dataSync();
        let inputShape = this.brain.inputWeights.shape;
        let outputShape = this.brain.outputWeights.shape;
        let newInWeighs = [];
        let newOutWeights = [];
        inputWeights.forEach(inputWeight => {
            if (random() <= mutateRate) {
                newInWeighs.push(inputWeight * random());
            } else {
                newInWeighs.push(inputWeight);
            }
        });
        outputWeights.forEach(outputWeight => {
            if (random() <= mutateRate) {
                newOutWeights.push(outputWeight * random());
            } else {
                newOutWeights.push(outputWeight);
            }
        });
        this.brain.inputWeights = tf.tensor(newInWeighs, inputShape);
        this.brain.outputWeights = tf.tensor(newOutWeights, outputShape);
    }

    draw() {
        if (!this.isAlive) {
            this.radius--;
            if (this.radius < 0) {
                this.radius = 0;
                noStroke();
            }
        } else {
            stroke(255, 0, 0);
            this.livingtime++;
            if (this.color.levels[2] == 200) {
                fill(100, 100, 255, 100);
                if (this.left) {
                    stroke(200, 200, 255);
                    ellipse(25, height - 25, 25);
                }
                if (this.right) {
                    stroke(200, 200, 255);
                    ellipse(55, height - 25, 25);
                }
                noStroke();
                text('L', 20, height - 20)
                text('R', 50, height - 20)
                ellipse(25, height - 25, 25);
                ellipse(55, height - 25, 25);
            }
        }
        if (this.color) {
            fill(this.color);
        } else {
            fill(255, 100);
        }
        ellipse(this.pos.x, this.pos.y, this.radius);
        line(this.pos.x, this.pos.y, this.pos.x + this.dir.x * this.radius / 2, this.pos.y + this.dir.y * this.radius / 2);
    }
}

class Plate {
    constructor(x, y, r) {
        this.pos = createVector(x, y);
        this.radius = r;
    }

    draw() {
        fill(220);
        noStroke();
        ellipse(this.pos.x, this.pos.y, this.radius * 2);
        fill(0);
        ellipse(this.pos.x, this.pos.y, this.radius);
    }

}