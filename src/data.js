// Particle: 
//  [[x,y],  # Position
//   [x, y], # Velocity }
//
//
import * as tf from '@tensorflow/tfjs';

export function newField(config) { 
  const w = config.width*config.density;
  const h = config.height*config.density;

  const dir = tf.randomUniform([w,h],0, 2 * Math.PI) 

  return dir.cos().stack(dir.sin(),2).print()
}

export function newParticles(config) {
  const posx = tf.randomUniform([config.particle_count], 0, config.width, 'float32')
  const posy = tf.randomUniform([config.particle_count], 0, config.height, 'float32')
  const vel  = tf.randomUniform([config.particle_count, 2], -1, 1, 'float32')

  const pos = posx.stack(posy,1);

  const particles = pos.stack(vel,1);
  particles.print()
   
  return particles;
}
