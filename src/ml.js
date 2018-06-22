import * as tf from '@tensorflow/tfjs';

const difference = 200;

/* Scoring functions */

export function closeToMiddle(points, config) {
  return tf.tidy(() => {
    const point = tf.tensor1d([config.width / 2, config.height / 2]);

    const diffs = points.squaredDifference(point);

    const t = 
      diffs.sum(1)
      .sub(tf.scalar(difference*difference)).square().softplus().mean();
    // t.print()

    return t;
  });
}


export function distanceTraveled(v1,v2) {
  return tf.tidy(() => {
    return v1.sub(v2).softplus().mean();
  });
}

export function distanceTraveledInverse(v1,v2) {
  return tf.tidy(() => {
    return v1.sub(v2).softplus().add(tf.scalar(-0.5)).mean();
  });
}


