import * as tf from '@tensorflow/tfjs'

export function randomOptimizer(fields, learningRate) {
  fields.print(true)

  const o = { 
    // confidence: tf.zerosLike(fields),
    confidence:   tf.onesLike(fields),
    values:       fields,
    learningRate: learningRate 
  }

  o.minimize = function (loss) {
    this.values = randomUpdate(this, tf.scalar(loss).clipByValue(0,1));
    this.confidence = this.confidence.mul(tf.scalar(1-this.learningRate)).clipByValue(0,1);
    // this.confidence.print()
  };

  return o;
}

function randomUpdate(model, loss) {
  const randomChange = tf.randomUniform(model.values.shape, -0.5, 0.5, 'float32');
  const actualChange = model.confidence.mul(randomChange).mul(loss);
  return model.values.add(actualChange);
}
