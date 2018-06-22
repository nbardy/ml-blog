import * as tf from '@tensorflow/tfjs'

export class DFOptimizer {
  // abstract minimize(f: () => tf.Scalar): null;
}

export class StocasticOptimizer extends DFOptimizer {
  constructor(learningRate, varList) {
    super();
    this.learningRate = learningRate;
    this.currentGrads = [];
    this.previousLoss = tf.scalar(Number.MAX_VALUE);
    this.entropy = 1;

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
    tf.tidy(() => {
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
        const change = 
          tf.randomUniform(variable.value.shape, -1, 1)
          .mul(tf.scalar(this.entropy))

        var newValue;

        if(lossChange.dataSync()[0] < 0) {
          newValue = value.add(change);
          this.previousLoss = tf.keep(loss);
        } else {
          newValue = value.sub(variable.lastChange).add(change);
        }

        variable.lastChange.assign(change)

        variable.value.assign(newValue);

        this.entropy = this.entropy * (1 - this.learningRate);
      }
    })
  }
}

export function stocastic(learningRate, varList) {
  return new StocasticOptimizer(learningRate, varList);
}

