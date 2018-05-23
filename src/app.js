import {newElement}            from '~/dom.js'
import {newField, newParticles, updateParticles}              
                               from '~/data.js'
import {drawParticles}         from '~/draw.js'
import css from '~/file.css';
import * as tf from '@tensorflow/tfjs'
import * as dat from 'dat.gui';



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

  function run(particles) {
    drawParticles(canvas, particles, config);
    const updatedParticles = updateParticles(particles, field, 1, config);
    requestAnimationFrame( function() { run(updatedParticles) })
  }

  run(particles)
}

const INITIAL_CONFIG = {
  width:   700,
  height:  700,
  density: 1/100,
  forceMagnitude: 13.67,
  initMagnitude: [0,0.4],
  velMagnitude: 1/10,
  maxVel: 20,
  particleCount: 10000
}

