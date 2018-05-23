import {newElement}            from '~/dom.js'
import {newField, newParticles, updateParticles}              
                               from '~/data.js'
import {drawParticles}         from '~/draw.js'
import css from '~/file.css';
import * as tf from '@tensorflow/tfjs'

import * as dat from 'dat.gui';

import {seededRandom} from '~/rand.js'

// console.log(seededRandom(4)())

console.log("--- Dev Mode ---")

var killPrevious = function() {};

function clean() {
  document.body.innerHTML = ""
  killPrevious()
}

function start(config) {
  // dt is amount of change in time 
  const dt = 1;
  const canvas = newElement("canvas", {width: config.width, height: config.height})

  // The force field
  const field = newField(config)
  // The particles of the simulation
  var particles = newParticles(config)

  drawParticles(canvas, particles, config)
  //
  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)

  // Use closure to kill
  var running = true;

  killPrevious = function() { running = false; }

  function run(particles) {
    drawParticles(canvas, particles, config);
    const updatedParticles = updateParticles(particles, field, 1, config);
    particles[0].dispose()
    particles[1].dispose()

    if(running) {
      requestAnimationFrame(
        function() { run(updatedParticles) })
    }
  }

  run(particles)
}


const DEV_CONFIG = {
  width:   700,
  height:  700,
  density: 1/100,
  initVelMagnitude: 12.1,
  initVelStdDev: 0.1,
  initForceMagnitude: 0,
  initForceStdDev: 2.1,
  resetRate: 0.01,
  forceMagnitude: 2,
  velMagnitude: 1/10,
  maximumVelocity: 23,
  particleCount: 5000,
  randomSeed: 50
}

function makeGUI() {
  const gui = new dat.GUI( { name: "Force Field" });
  gui.add(DEV_CONFIG, "forceMagnitude", 0,10)
  gui.add(DEV_CONFIG, "velMagnitude", 0,1)
  gui.add(DEV_CONFIG, "maximumVelocity", 0, 60)
  // gui.add(DEV_CONFIG, "randomSeed", 0, 100, 1)
}

if(module.hot) {
  clean()
  makeGUI()
  module.hot.accept();
  start(DEV_CONFIG)
}
