import {newElement}            from '~/dom.js'
import {newField, newParticles}              
                               from '~/data.js'

// console.log(Igloo)

console.log("Dev mode")

function clean() {
  document.body.innerHTML = ""
}

function start(config) {
  // dt is amount of change in time 
  const dt = 1;
  const canvas = newElement("canvas", {width: config.width, height: config.height})

  // The force field
  const field = newField(config)
  // The particles of the simulation
  const particles = newParticles(config)

  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)
}

var DEV_CONFIG = {
  width:   400,
  height:  400,
  density: 1/10,
  particle_count: 4
}

if(module.hot) {
  console.log(module.hot);
  clean()
  module.hot.accept();
  start(DEV_CONFIG)
}
