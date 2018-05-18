import {newElement}            from '~/dom.js'
import {newField, newParticles, updateParticles}              
                               from '~/data.js'
import {drawParticles}         from '~/draw.js'

import css from '~/file.css';


console.log("Dev mode")

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

    if(running) {
      requestAnimationFrame(
        function() { run(updatedParticles) })
    }
  }

  run(particles)
}

//TODO: Add wrap to particles
//
const DEV_CONFIG = {
  width:   400,
  height:  400,
  density: 1,
  forceMagnitude: 1/10,
  velMagnitude: 6,
  particleCount: 10000
}

if(module.hot) {
  console.log(module.hot);
  clean()
  module.hot.accept();
  start(DEV_CONFIG)
}
