import * as tf from '@tensorflow/tfjs'
import * as data from '~/data'


export class ModelOptimizer {

  initialModel(variable) {
    const input = tf.input({shape: variable.shape});

    variable.print(true)
    // I don't know what I'm doing
    // Second dense layer uses softmax activation.
    const denseLayer2 = tf.layers.dense({units: 1, activation: 'softmax', useBias: true});
    const flatten = tf.layers.flatten()
    const final = tf.layers.dense({units: 1, activation: 'relu6'})

    // Obtain the output symbolic tensor by applying the layers on the input.
    // TODO: Do some research on what model this should be.
    const output = final.apply(flatten.apply(denseLayer2.apply(input)));
    // const output = final.apply(flatten.apply(denseLayer2.apply(lstmLayer1.apply(denseLayer1.apply(input)))));

    // Create the model based on the inputs.
    const model = tf.model({inputs: input, outputs: output});

    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

    return model;
  }

  // TODO Add boundary option here in config.
  constructor(varList, boundList, config) {
    this.learningRate = config.learningRate;
    this.entropyDecay = config.entropyDecay;
    this.currentGrads = [];
    this.entropy = 1;
    this.searchSize = config.searchSize;
    this.epochs = config.epochs
    this.boundList =  boundList
    if(varList.length !== boundList.length) {
      throw new Error("List of bounds must match the size of the list of variables.")
    }

    for(let i = 0; i < varList.length; i++) { 
      let variable = varList[i],
          bounds   = boundList[i];

      const model = this.initialModel(variable);
      const initialPrediction = tf.unstack(model.predict(tf.stack([variable])))[0];

      initialPrediction.print()

      this.currentGrads.push({
        model: model,
        prediction: initialPrediction,
        // Direction and momentum start at zero.
        momentum: tf.zerosLike(variable),
        bounds: bounds,
        value: variable
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
        const {model, velocity, value, bounds} = variable;
        const valueKeep = tf.keep(tf.stack([value]));

        // TODO: Remove this
        // NOTE: This handle the firs iteration with no predictions, make that prediction with initial data.
        const predictionError = tf.keep(tf.stack([variable.prediction.sub(loss)]));

        // Fit Loss prediction model with last generation
        const h = model.fit(valueKeep, predictionError, { batchSize: this.epochs, epochs: this.epochs });
        // h.then(function(_) { loss.dispose() });

        const potentialNewValues = [];
        for(let i = 0; i < this.searchSize; i++) {
          const change = 
            tf.randomNormal(
              variable.value.shape, 
              (bounds[1] + bounds[0]) / 2,
              (bounds[1] - bounds[0]) / 2) // NOTE: Need better way to calculate std
            .mul(tf.scalar(this.entropy))

          const newValue = value.add(change).clipByValue(bounds[0],bounds[1])

          potentialNewValues.push(newValue);
        }

        // TODO: Try to remove dataSync call.
        // const newValuesTensor = tf.tensor(potentialNewValues);
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

export function modelOptimizer(learningRate, bounds, varList) {
  return new ModelOptimizer(learningRate, bounds, varList);
}

