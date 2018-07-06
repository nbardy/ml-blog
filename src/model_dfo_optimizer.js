import * as tf from '@tensorflow/tfjs'


export class ModelOptimizer {

  // initialModel(variable) {
    // return model;
  // }

  initialModel(variable) {
    const input = tf.input({shape: variable.shape});

    variable.print(true)
    // First dense layer uses relu activation.
    const denseLayer1 = tf.layers.dense({units: 4, activation: 'relu'});
    // I don't know what I'm doing
    const lstmLayer1 = tf.layers.lstm({units: 2, returnSequences: true});
    // Second dense layer uses softmax activation.
    const denseLayer2 = tf.layers.dense({units: 1, activation: 'softmax', useBias: true});
    const flatten = tf.layers.flatten()
    const final = tf.layers.dense({units: 1, activation: 'relu6'})

    // Obtain the output symbolic tensor by applying the layers on the input.
    // TODO: Do some research on what model this should be.
    const output = final.apply(flatten.apply(denseLayer2.apply(lstmLayer1.apply(denseLayer1.apply(input)))));

    // Create the model based on the inputs.
    const model = tf.model({inputs: input, outputs: output});

    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

    return model;
  }

  constructor(varList, config) {
    this.learningRate = config.learningRate;
    this.entropyDecay = config.entropyDecay;
    this.currentGrads = [];
    this.entropy = 1;
    this.searchSize = config.searchSize;
    this.epochs = config.epochs

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
      const actualLoss = tf.keep(tf.stack([f()]));

      // Update Each Variable
      for(var variable of this.currentGrads) {
        // Conform and Persist values
        const {model, velocity, value} = variable;
        const valueKeep = tf.keep(tf.stack([value]));

        // TODO: Remove this
        // NOTE: This handle the firs iteration with no predictions, make that prediction with initial data.
        if(variable.prediction) {
        const predictionError = tf.keep(tf.stack([variable.prediction.sub(loss)]));

        // Fit Loss prediction model with last generation
        const h = model.fit(valueKeep, predictionError, { batchSize: 4, epochs: this.epochs });
          // h.then(function(_) { loss.dispose() });
          }

        const potentialNewValues = [];
        for(let i = 0; i < this.searchSize; i++) {
          const change = 
            tf.randomUniform(variable.value.shape, -1, 1)
            .mul(tf.scalar(this.learningRate))

          potentialNewValues.push(value.add(change))
        }

        const predictions = model.predict(tf.stack(potentialNewValues))
        const newValue = potentialNewValues[predictions.argMax().dataSync()[0]]
        const predictionForNewValue = predictions.max();

        variable.value.assign(newValue);
        variable.prediction = tf.keep(predictionForNewValue);

        this.entropy = this.entropy * this.entropyDecay;
        return loss;
      }
    })
  }
}

export function modelOptimizer(learningRate, varList) {
  return new ModelOptimizer(learningRate, varList);
}

