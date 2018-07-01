import * as tf from '@tensorflow/tfjs'


export class ModelOptimizer {

  // initialModel(variable) {
    // const input = tf.input({shape: variable.shape});

    // // First dense layer uses relu activation.
    // const denseLayer1 = tf.layers.dense({units: variable.length, activation: 'relu'});
    // // Second dense layer uses softmax activation.
    // const denseLayer2 = tf.layers.dense({units: 1, activation: 'softmax'});

    // // Obtain the output symbolic tensor by applying the layers on the input.
    // const output = denseLayer2.apply(denseLayer1.apply(input));

    // // Create the model based on the inputs.
    // const model = tf.model({inputs: input, outputs: output});

    // model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})
    // return model;
  // }

  initialModel(variable) {
    const model = tf.sequential();

    // First layer must have an input shape defined.
    model.add(tf.layers.dense({units: 32, inputShape: variable.shape}));
    // Afterwards, TF.js does automatic shape inference.
    model.add(tf.layers.dense({units: 4}));

    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

    return model;
  }

  constructor(varList, config) {
    this.learningRate = config.learningRate;
    this.entropyDecay = config.entropyDecay;
    this.currentGrads = [];
    this.previousLoss = tf.scalar(Number.MAX_VALUE);
    this.entropy = 1;

    for(var variable of varList) {
      this.currentGrads.push({
        model: this.initialModel(variable),
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
    return tf.tidy(() => {
      // If there is no previous result calucate loss and store
      const loss = f();

      var lossChange;

      // 0 lossChange is good, 1 loss change is bad
      lossChange = loss.sub(this.previousLoss)

      for(var variable of this.currentGrads) {
        const {model, velocity, value} = variable;
        const h = model.fit(tf.stack([value]), tf.stack([loss]), 
          {
            batchSize: 4,
            epochs: 3
          })

        // for(let i = 0; i < 10; i++) {
          const change = 
          tf.randomUniform(variable.value.shape, -1, 1)
          .mul(tf.scalar(this.learningRate))
        // }


        var newValue;

        // TODO: Speed this up.
        if(lossChange.dataSync()[0] < 0) {
          newValue = value.add(change);
          this.previousLoss = tf.keep(loss);
        } else {
          newValue = value.sub(variable.lastChange).add(change);
        }

        variable.lastChange.assign(change)

        variable.value.assign(newValue);

        this.entropy = this.entropy * this.entropyDecay;

        return loss;
      }
    })
  }
}

export function modelOptimizer(learningRate, varList) {
  return new ModelOptimizer(learningRate, varList);
}

