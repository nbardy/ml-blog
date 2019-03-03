// Particle:
//  [[x,y],  # Position
//   [x, y], # Velocity }
//
//
import * as tf from '@tensorflow/tfjs';
import '~/helpers.js';

export function newField(config) {
  const w = config.width * config.density;
  const h = config.height * config.density;

  return tf.tidy(() => {
    const dir = tf.randomUniform([w * h], 0, 2 * Math.PI);

    const mag = tf.randomNormal(
      [w * h],
      config.initForceMagnitude,
      config.initForceStdDev,
      'float32',
      config.randomSeed,
    );

    return dir
      .cos()
      .mul(mag)
      .stack(dir.sin().mul(mag), 1);
  });
}

export function newParticles(config) {
  return tf.tidy(() => {
    const posx = tf.randomUniform(
      [config.particleCount],
      0,
      config.width,
      'float32',
    );
    const posy = tf.randomUniform(
      [config.particleCount],
      0,
      config.height,
      'float32',
    );
    const pos = posx.stack(posy, 1);
    const vel = tf.zerosLike(pos);

    return [pos, vel];
  });
}

export function clipField(field, mag) {
  return tf.tidy(() => {
    return field.clipByValue(-mag, mag);
  });
}

export function updateParticles2([pos, vel], model, dt, generation, config) {
  return tf.tidy(() => {
    // Scale down to fit force field dimensions
    const particles = [pos, vel];

    const posNormalized = pos.div(
      tf.tensor2d([config.width, config.height], [1, 2]),
    );

    const velNormalized = vel.div(tf.scalar(config.maximumVelocity));

    const axis = 1;

    const forces = model.predict(
      tf.concat([posNormalized, velNormalized], axis),
    );

    // Shift forces from 0,1 to -0.5,0.5
    const forcesShifted = forces.sub(tf.scalar(0.5));
    if (generation % config.printRate == 0) {
      forcesShifted.print();
    }

    // Forces applied with relevant magnitude
    const forcesScaled = forcesShifted.mul(tf.scalar(config.forceMagnitude));

    const updatedVel = vel.add(forcesScaled).mul(tf.scalar(config.friction));
    const updatedPos = pos.add(updatedVel);

    // Wrap Positions
    const posX = updatedPos.slice([0, 0], [-1, 1]).mod(tf.scalar(config.width));

    const posY = updatedPos
      .slice([0, 1], [-1, 1])
      .mod(tf.scalar(config.height));

    // Cap Vels
    const velX = updatedVel
      .slice([0, 0], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    const velY = updatedVel
      .slice([0, 1], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    const updateVelCapped = velX.concat(velY, 1);
    const updatePosWrapped = posX.concat(posY, 1);

    return [updatePosWrapped, updateVelCapped];
  });
}

export function updateParticles([pos, vel], field, dt, config) {
  return tf.tidy(() => {
    // Scale down to fit force field dimensions
    const scaled = tf.floor(pos.mul(tf.scalar(config.density))).toInt();

    // No gather_nd in tgjs so things must be flattened from x,y => i index
    const indices = scaled.xytoI(config.width * config.density);

    // All of the force which should effect each particle
    const forces = field
      .sigmoid()
      .sub(tf.scalar(0.5))
      .gather(indices);

    // Forces applied with relevant magnitude
    const forcesScaled = forces.mul(tf.scalar(config.forceMagnitude));

    const updatedVel = vel.add(forcesScaled).mul(tf.scalar(config.friction));
    const updatedPos = pos.add(updatedVel);

    // Wrap Positions
    const posX = updatedPos.slice([0, 0], [-1, 1]).mod(tf.scalar(config.width));

    const posY = updatedPos
      .slice([0, 1], [-1, 1])
      .mod(tf.scalar(config.height));

    // Cap Vels
    const velX = updatedVel
      .slice([0, 0], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    const velY = updatedVel
      .slice([0, 1], [-1, 1])
      .clipByValue(-config.maximumVelocity, config.maximumVelocity);

    const updatePosWrapped = posX.concat(posY, 1);
    const updateVelCapped = velX.concat(velY, 1);

    return [updatePosWrapped, updateVelCapped];
  });
}
