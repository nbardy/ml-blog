import * as tf from '@tensorflow/tfjs';

export function newModel(config, version) {
  const f = eval("newModel" + version );
  return f(config);
}

export function newModel1(config) { 
  const w = config.width*config.density;
  const h = config.height*config.density;

  const model2 = tf.sequential();


  const pos = tf.input({shape: [2]});
  const vel = tf.input({shape: [2]});

  const concatLayer = tf.layers.concatenate();

  const layer0 = concatLayer.apply([pos,vel]);
  const layer1 = tf.layers.dense({units: 16});
  const act2 = tf.layers.activation({activation: 'selu'});
  const layer2 = tf.layers.dense({units: (w*h)*4});
  const act3 = tf.layers.activation({activation: 'softmax'});
  const layerdropout = tf.layers.dropout({rate: 0.6})
  const layer4 = tf.layers.dense ({units: 2});
  const layer5 = tf.layers.activation({activation: 'sigmoid'});

  //IDEA: Make another network that tries to predict. Than train it be unpredictable


  // Use tf.layers.input() to obtain a SymbolicTensor as input to apply().
  const output1 = 
    layer5.apply(
      layerdropout.apply(
        layer4.apply(
          // layer3.apply(
          act3.apply(
          layer2.apply(
            act2.apply(
              layer1.apply(
                layer0
              )))))))
// ))));


  const model = tf.model({inputs: [pos, vel], outputs: output1});


  return model

}

// Speed model
export function newModel2(config) { 
  const w = config.width*config.density;
  const h = config.height*config.density;

  const model2 = tf.sequential();
  model2.add(tf.layers.dense({units: 4, inputShape: [4]}))
  model2.add(tf.layers.dense({units: 20}))
  model2.add(tf.layers.dropout({rate: 0.6}))
  model2.add(tf.layers.dense({units: 8}))
  model2.add(tf.layers.dropout({rate: 0.2}))
  model2.add(tf.layers.dense({units: 2, outputShape: [2]}))
  model2.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  return model2;

}


//  Add relu
export function newModel3(config) { 
  const w = config.width*config.density;
  const h = config.height*config.density;

  const model2 = tf.sequential();
  model2.add(tf.layers.dense({units: 4, inputShape: [4]}))
  model2.add(tf.layers.dense({units: 20}))
  model2.add(tf.layers.dense({units: 100, activation: 'relu6'}))
  model2.add(tf.layers.dropout({rate: 0.6}))
  model2.add(tf.layers.dense({units: 8}))
  model2.add(tf.layers.dropout({rate: 0.2}))
  model2.add(tf.layers.dense({units: 2, outputShape: [2]}))
  model2.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  return model2;

}
