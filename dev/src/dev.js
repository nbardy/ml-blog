import {newElement}            from '~/dom.js'
import {newField, newParticles, updateParticles }              
                               from '~/data.js'
import {closeToMiddle, distanceTraveled}         
                               from '~/ml.js'
import {drawScene}         from '~/draw.js'
import css from '~/file.css';

import * as tf from '@tensorflow/tfjs'

import * as dat from 'dat.gui';

import {seededRandom} from '~/rand.js'
import * as learn from '~/learn.js'
import * as dfo from '~/stocastic_dfo_optimizer.js'


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
  const canvas = newElement("canvas", {width: config.width, height: config.height})

  // The force field
  const field = tf.variable(newField(config));
  // The particles of the simulation
  var particles = newParticles(config)

  // const optimizer = learn.randomOptimizer(field, 0.001)
  const optimizer = dfo.stocastic(config.learningRate, [field]);

  drawScene(canvas, particles, field, config)
  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)

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
    // console.log("particles")
    // particles[0].print()
    // for(var i = 0; i < config.updatesPerOptimizer; i++) {
    const updatedParticles = 
      updateParticles(particles, field, 1, config);
    generation++;

    if((generation % config.drawRate) == 0) {
      drawScene(canvas, updatedParticles, field, config);
    }




    /* Memory Cleanup */
    // if((generation % config.sampleRate) == 0) {
    //   storedStates.push(particles);

    //   if((generation % config.trainRate) == 0) {

    //     optimizer.minimize(() => {
    //       return tf.tidy(() =>  {
    //         const dist = distanceTraveled(particles[0], updatedParticles[0])

    //         const val = closeToMiddle(updatedParticles[0], config);
    //         console.log("U")
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
      particles[0].dispose();
      particles[1].dispose();
    // }

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
  density: 1/50,
  initVelMagnitude: 12.1,
  initVelStdDev: 0.1,
  initForceMagnitude: 0,
  initForceStdDev: 5.1,
  resetRate: 0.01,
  forceMagnitude: 1.2,
  friction: 0.941,
  maximumVelocity: 5,
  maximumForce: 15,
  particleCount: 300,
  learningRate: 0.00011,
  updatesPerOptimizer: 1,
  drawRate: 1,
  sampleRate: 25,
  trainRate: 50,
  randomSeed: 50,
  drawField: false
}

function makeGUI() {
  const gui = new dat.GUI( { name: "Force Field" });
  gui.add(DEV_CONFIG, "forceMagnitude", 0,10)
  gui.add(DEV_CONFIG, "friction", 0,1)
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
