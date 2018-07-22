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

const distance =  300;

// Returns a tf.scalar result from 0 -> 1 indicating the percent of points in the desired area
export function percentInZone(points, config) {
  return tf.tidy(() => {
    const point = tf.tensor1d([config.width / 2, config.height / 2]);

    const diffs = points.squaredDifference(point);

    const percent = 
      diffs.sum(1).sqrt().less(tf.scalar(distance)).toFloat().sum().div(tf.scalar(config.particleCount))
    // t.print()

    return tf.scalar(1).sub(percent);
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


