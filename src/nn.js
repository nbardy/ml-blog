import  * as dl from 'deeplearn';

console.log(dl)
const a = dl.variable(dl.scalar(Math.random()));
const b = dl.variable(dl.scalar(Math.random()));
const c = dl.variable(dl.scalar(Math.random()));

const learningRate = 0.01;
const optimizer = dl.train.sgd(learningRate);

/*
 * This function represents our 'model'. Given an input 'x' it will try and predict
 * the appropriate output 'y'.
 *
 * This could be as complicated a 'neural net' as we would like, but we can just
 * directly model the quadratic equation we are trying to model.
 *
 * It is also sometimes referred to as the 'forward' step of our training process.
 * Though we will use the same function for predictions later.
 *
 *
 * @return number predicted y value
 */function predict(input) {
  // y = a * x ^ 2 + b * x + c
  return dl.tidy(() => {
    const x = dl.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}
/*
 * This will tell us how good the 'prediction' is given what we actually expected.
 *
 * prediction is a tensor with our predicted y value.
 * actual number is a number with the y value the model should have predicted.
 */
function loss(prediction, actual) {
  // Having a good error metric is key for training a machine learning model
  const error = dl.scalar(actual).sub(prediction).square();
  return error;
}

/*
 * This will iteratively train our model. We test how well it is doing
 * after numIterations by calculating the mean error over all the given
 * samples after our training.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations, done) {
  let currentIteration = 0;

  for (let iter = 0; iter < numIterations; iter++) {
    for (let i = 0; i < xs.length; i++) {
      // Minimize is where the magic happens, we must return a
      // numerical estimate (i.e. loss) of how well we are doing using the
      // current state of the variables we created at the start.

      // This optimizer does the 'backward' step of our training data
      // updating variables defined previously in order to minimize the
      // loss.
      optimizer.minimize(() => {
        // Feed the examples into the model
        const pred = predict(xs[i]);
        const predLoss = loss(pred, ys[i]);

        return predLoss;
      });
    }

    // Use dl.nextFrame to not block the browser.
    await dl.nextFrame();
  }

  done();
}

/*
 * This function compare expected results with the predicted results from
 * our model.
 */
function test(xs, ys) {
  dl.tidy(() => {
    const predictedYs = xs.map(predict);
    console.log('Expected', ys);
    console.log('Got', predictedYs.map((p) => p.dataSync()[0]));
  })
}

const data = {
  xs: [0, 1, 2, 3],
  ys: [1.1, 5.9, 16.8, 33.9]
};

function runml() {
  // Lets see how it does before training.
  console.log('Before training: using random coefficients')
  test(data.xs, data.ys);
  train(data.xs, data.ys, 500, () => {
    console.log(`After training: a=${a.dataSync()}, b=${b.dataSync()}, c=${c.dataSync()}`)
    test(data.xs, data.ys);
  });
}

export {runml}
