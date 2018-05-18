// Particle: 
//  [[x,y],  # Position
//   [x, y], # Velocity }
//
//
import * as tf from '@tensorflow/tfjs';
import '~/helpers.js';

export function newField(config) { 
  const w = config.width*config.density;
  const h = config.height*config.density;

  const dir = tf.randomUniform([w*h],0, 2 * Math.PI);

  return dir.cos().stack(dir.sin(),1)
}

export function newParticles(config) {
  const posx = tf.randomUniform([config.particleCount], 0, config.width, 'float32')
  const posy = tf.randomUniform([config.particleCount], 0, config.height, 'float32')
  const vel  = tf.randomUniform([config.particleCount, 2], -1, 1, 'float32')

  const pos = posx.stack(posy,1);
   
  return [pos,vel];
}

export function updateParticles([pos, vel],field, dt, config) {
  // Scale down to fit force field dimensions
  const scaled = tf.round(pos.mul(tf.scalar(config.density))).toInt()

  // No gather_nd in tgjs so things must be flattened from x,y => i index
  const indices = scaled.xytoI(config.width);

  // All of the force which should effect each particle
  const forces = field.gather(indices);
  
  // Forces applied with relevant magnitude
  const forcesScaled = forces.mul(tf.scalar(config.forceMagnitude));

  const updatedVel = vel.add(forcesScaled)
  const updatedPos = pos.add(updatedVel.mul(tf.scalar(config.velMagnitude)))


  return [updatedPos, updatedVel];
}

