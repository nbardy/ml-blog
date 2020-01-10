import { newElement } from "~/dom.js";
import {
  clipField,
  newField,
  newParticles,
  updateParticles,
  updateParticles2
} from "~/data.js";

import { newModel } from "~/models.js";

import { closeToMiddle, percentInZone, distanceTraveled } from "~/loss.js";

import { drawScene } from "~/draw.js";
import css from "~/file.css";

import * as tf from "@tensorflow/tfjs";

import * as dat from "dat.gui";
import { seededRandom } from "~/rand.js";
import * as learn from "~/learn.js";
import * as chart from "~/chart_optimizer.js";
import * as sdfo from "~/stocastic_dfo_optimizer.js";
import * as mdfo from "~/model_dfo_optimizer.js";

console.log("--- Dev Mode ---");

function clearStates(states) {
  while (states.length > 0) {
    let [p, v] = states.pop();
    p.dispose();
    v.dispose();
  }
}

function start(config) {
  // dt is amount of change in time
  const dt = 1;
  const chart_data = [];
  const canvas = newElement("canvas", {
    width: window.innerWidth,
    height: window.innerHeight
  });
  const canvasChart = newElement("canvas", { width: 500, height: 300 });

  // The force field
  const model = newModel(config, 3);
  const field = tf.variable(newField(config));
  // The particles of the simulation
  var initialParticles = newParticles(config);

  // I was trying to use non differentiable optimizers
  // I ended up swapping the update function to be differentiable
  //
  // const optimizer = learn.randomOptimizer(field, 0.001)
  //
  // const optimizer = mdfo.optimizer(
  //   [field],
  //   [[-15,15]],
  //   config
  // );

  // const optimizer = sdfo.optimizer([field], config)

  // TODO; Change from trackOptimizer, to postData
  // chart.trackOptimizer(optimizer, canvasChart)
  //
  let optimizer = tf.train.adam(config.learningRate);

  // Draw first Scene
  drawScene(canvas, initialParticles, field, config);

  // Create needed dom elements
  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas);
  document.body.appendChild(canvasChart);

  // Use closure to kill
  var running = true;

  var updatedParticles;
  var nextParticleState;
  var keptParticles;
  var generation = 0;

  const storedStates = [];
  var p1;
  var p2;
  var counter;

  let current_run_id = 0;
  function refresh() {
    // Reset optimizer
    optimizer = tf.train.adam(config.learningRate);
    const ps = newParticles(config);
    current_run_id = current_run_id + 1;
    run(ps, current_run_id);
  }

  function run(particles, run_id) {
    optimizer.minimize(() => {
      // particles[0].print()
      // for(var i = 0; i < config.updatesPerOptimizer; i++) {
      updatedParticles =
        // updateParticles(particles, field, 1, config);
        updateParticles2(particles, model, 1, generation, config);

      keptParticles = updatedParticles.map(tf.keep);

      generation++;

      if (generation % config.drawRate == 0) {
        window.requestAnimationFrame(() => {
          drawScene(canvas, updatedParticles, field, config);
        });
      }

      const val = closeToMiddle(updatedParticles[0].slice(0, 200), config);
      return val;
    });

    /* Multi Frame Cleanup Code */
    // if((generation % config.sampleRate) == 0) {
    //   storedStates.push(particles);

    //   if((generation % config.trainRate) == 0) {

    //     optimizer.minimize(() => {
    //       return tf.tidy(() =>  {
    //         const dist = distanceTraveled(particles[0], updatedParticles[0])

    //         const val = closeToMiddle(updatedParticles[0], config);
    //         updatedParticles[0].print()
    //         const val2 = closeToMiddle(updatedParticles[1], config);
    //         val.print()
    //         val2.print()
    //         return val.add(val2);
    //       });
    //     });

    //     clearStates(storedStates);
    //   }

    // } else {
    //
    // //
    //
    // * Single Frame Cleanup Code *//
    particles[0].dispose();
    particles[1].dispose();

    console.log("numTensors : " + tf.memory().numTensors);
    // }

    if (run_id === current_run_id) {
      setTimeout(function() {
        run(keptParticles, run_id);
      });
    }
  }

  run(initialParticles, current_run_id);
}

window.tf = tf;

const DEV_CONFIG = {
  width: window.innerWidth,
  height: window.innerHeight,
  density: 1 / 50,
  initVelMagnitude: 8.1,
  initVelStdDev: 0.1,
  initForceMagnitude: 0,
  initForceStdDev: 5.1,
  resetRate: 0.05,
  alphaBlend: 0.13,
  forceMagnitude: 1.51,
  friction: 0.311,
  maximumVelocity: 12.2,
  maximumForce: 13.2,
  particleCount: 1000,
  learningRate: 0.01,
  entropyDecay: 0.99,
  updatesPerOptimizer: 1,
  drawRate: 1,
  sampleRate: 1,
  printRate: 40,
  trainRate: 20,
  randomSeed: 50,
  searchSize: 20,
  epochs: 3,
  drawField: false,
  clip: i => clipField(i, 15),
  backgroundColor: [12, 0, 34]
};

// Others
// Contnious
// fricction: 0.987 More cot
// maxVel : 12.9
// forceMag 1

function makeGUI() {
  const gui = new dat.GUI({ name: "Force Field" });
  gui.add(DEV_CONFIG, "resetRate", 0, 1);
  gui.add(DEV_CONFIG, "alphaBlend", 0, 1);
  gui.add(DEV_CONFIG, "drawField");
  gui.add(DEV_CONFIG, "forceMagnitude", 0, 10);
  gui.add(DEV_CONFIG, "friction", 0.5, 1);
  gui.add(DEV_CONFIG, "maximumVelocity", 0, 60);
  gui.add(DEV_CONFIG, "drawRate", 0, 200, 1);
  gui.add(DEV_CONFIG, "density", 1 / 2000, 1);
  gui.add(DEV_CONFIG, "particleCount");
  // gui.add(DEV_CONFIG, "randomSeed", 0, 100, 1)
}

console.log("debug/pre-hot");

if (module.hot) {
  makeGUI();

  const app = start(DEV_CONFIG);
  console.log("debug/re");
  module.hot.accept("./dev.js", () => {
    console.log("debug/re2");
    app.refresh();
  });
}
