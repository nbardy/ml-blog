import * as tf from '@tensorflow/tfjs';

export function closeToMiddle(points, config) {
  const point = tf.tensor1d([config.width / 2, config.height / 2]);

  return points.sub(point).square().sum().mean();
}
