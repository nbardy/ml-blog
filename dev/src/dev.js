import {newElement}            from '~/dom.js'
import {newField, newParticles, updateParticles }              
                               from '~/data.js'
import {closeToMiddle, percentInZone, distanceTraveled}         
                               from '~/loss.js'
import {drawScene}         from '~/draw.js'
import css from '~/file.css';

import * as tf from '@tensorflow/tfjs'

import * as dat from 'dat.gui';

import {seededRandom} from '~/rand.js'
import * as learn from '~/learn.js'
import * as chart from '~/chart_optimizer.js'
import * as sdfo from '~/stocastic_dfo_optimizer.js'
import * as mdfo from '~/model_dfo_optimizer.js'


// console.log(seededRandom(4)())

console.log("--- Dev Mode ---")

var killPrevious = function() {};

function clean() {
  document.body.innerHTML = ""
  killPrevious()
}

function clearStates(states) {
  while(states.length > 0) {
    let [p,v] = states.pop();
    p.dispose();
    v.dispose();
  }
}

function start(config) {
  // dt is amount of change in time 
  const dt = 1;
  const chart_data = [];
  const canvas = newElement("canvas", {width: config.width, height: config.height})
  const canvasChart = newElement("canvas", {width: config.width, height: config.height})

  // The force field
  const field = tf.variable(newField(config));
  // The particles of the simulation
  var particles = newParticles(config)

  // const optimizer = learn.randomOptimizer(field, 0.001)
  const optimizer = mdfo.modelOptimizer([field], config);

  chart.trackOptimizer(optimizer, canvasChart)

  drawScene(canvas, particles, field, config)
  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)
  document.body.appendChild(canvasChart)

  // Use closure to kill
  var running = true;

  killPrevious = function() { running = false; }

  var updatedParticles;
  var nextParticleState;
  var generation = 0;

  const storedStates = [];
  var p1;
  var p2;
  var counter;

  function run(particles) {
    // for(var i = 0; i < config.updatesPerOptimizer; i++) {
    const updatedParticles = 
        updateParticles(particles, field, 1, config);

    generation++;

    if((generation % config.drawRate) == 0) {
      drawScene(canvas, updatedParticles, field, config);
    }

    if((generation % config.trainRate) == 0) {
      optimizer.minimize(() => {
        const val = percentInZone(updatedParticles[0], config);
        return val;
      });
    }

    // return val.add(val2);
    // const dist = distanceTraveled(particles[0], updatedParticles[0])
    //


    /* Memory Cleanup */
    if((generation % config.sampleRate) == 0) {
      storedStates.push(particles);

      if((generation % config.trainRate) == 0) {
        clearStates(storedStates);
      }

    } else {
      particles[0].dispose();
      particles[1].dispose();
    }

    if(running) {
      requestAnimationFrame(
        function() { run(updatedParticles) })
    }
  }

  run(particles)
}

window.tf = tf;

const DEV_CONFIG = {
  width:   500,
  height:  500,
  density: 1/100,
  initVelMagnitude: 12.1,
  initVelStdDev: 0.1,
  initForceMagnitude: 0,
  initForceStdDev: 5.1,
  resetRate: 0.01,
  forceMagnitude: 3.2,
  momentum: 0.141,
  maximumVelocity: 22,
  maximumForce: 15,
  particleCount: 4000,
  learningRate: 1.41,
  entropyDecay: 0.99,
  updatesPerOptimizer: 1,
  drawRate: 1,
  sampleRate: 1,
  trainRate: 200,
  randomSeed: 50,
  drawField: true
}

function makeGUI() {
  const gui = new dat.GUI( { name: "Force Field" });
  gui.add(DEV_CONFIG, "forceMagnitude", 0,10)
  gui.add(DEV_CONFIG, "momentum", 0,1)
  gui.add(DEV_CONFIG, "maximumVelocity", 0, 60)
  gui.add(DEV_CONFIG, "drawRate", 0, 200, 1);
  gui.add(DEV_CONFIG, "drawField");
  // gui.add(DEV_CONFIG, "randomSeed", 0, 100, 1)
}



if(module.hot) {
  clean()

  makeGUI()
  module.hot.accept();
  start(DEV_CONFIG)
}
