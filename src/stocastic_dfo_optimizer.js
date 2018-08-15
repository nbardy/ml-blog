import * as tf from '@tensorflow/tfjs'


export class DFOptimizer {
  // abstract minimize(f: () => tf.Scalar): null;
}

export class StocasticOptimizer extends DFOptimizer {
  constructor(varList, config) {
    super();
    this.learningRate = config.learningRate;
    this.entropyDecay = config.entropyDecay;
    this.currentGrads = [];
    this.previousLoss = tf.scalar(Number.MAX_VALUE);
    this.entropy = 1;
    this.clip = config.clip;

    for(var variable of varList) {
      this.currentGrads.push({
        // Direction and momentum start at zero.
        value: variable,
        lastChange:  tf.variable(tf.zerosLike(variable))
      })
    }
  }

  dispose() {
    this.previousLoss.dispose()
    for(var grad of this.currentGrads) {
      grad.lastChange.dispose();
    }
  }

  // f is the loss function. minimize tries to bring loss result closer to 0
  // TODO: Add tidy/dipose
  minimize(f) {
    const clip = this.clip;
    return tf.tidy(() => {
      // If there is no previous result calucate loss and store
      const loss = f();
      var lossChange;

      // 0 lossChange is good, 1 loss change is bad
      lossChange = loss.sub(this.previousLoss)

      for(var variable of this.currentGrads) {
        const {velocity, value} = variable;

        //  I should update the velocity based on resultDifference and momentum  and entropy.
        //  I should 
        // Result has improved
        const perChange = 
          tf.randomNormal(variable.value.shape, 0, this.learningRate * this.entropy)

        const change = value.mul(perChange)

        var newValue;
        const randomChance = Math.random();

        if((randomChance < this.entropy) || (lossChange.dataSync()[0] < 0)) {
          newValue = value.add(change);
          this.previousLoss = tf.keep(loss);
        } else {
          newValue = value.sub(variable.lastChange).add(perChange);
        }

        variable.lastChange.assign(change)

        const newValueClipped = clip ? clip(newValue) : newValue 
        newValueClipped.print()

        variable.value.assign(newValueClipped);

        this.entropy = this.entropy * this.entropyDecay;
      }

      return loss;
    })
  }
}

export function optimizer(varList, config) {
  return new StocasticOptimizer(varList, config);
}

