import '~/helpers.js';
import * as tf from '@tensorflow/tfjs';


export function drawParticles(canvas, [posTensor, velTensor] , config) {
  const {width,height} = config;
  const bytes = new Uint8ClampedArray(width * height * 4); 

  const velScale = 2 * config.maxVel*config.maxVel

  // TODO use tensorflow to parrellize colors
  tf.tidy(() => {
    // const positions  = posTensor.data();
    // const velocities = velTensor.data();

    const particleCount = posTensor.shape[0]

    // Transform to I index equivlanet for imageBuffer insertion
    const posIdx = posTensor.xytoI(width).mul(tf.scalar(4, 'int32'));
    // Transform to a magnitude value to be used as color
    const colors =  velTensor.div(tf.scalar(config.maximumVelocity))

    for(let i = 0, imgIndex, red, green; i < posIdx.size; i++) {
      imgIndex = posIdx.get(i)
      red      = colors.get(2*i)
      green      = colors.get(2*i + 1)

      bytes[imgIndex] = 125 + 125 * red 
      bytes[imgIndex + 2] = 125 + 125 * green;
      bytes[imgIndex + 3] = 255;
    }

    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(bytes, width, height)
    ctx.putImageData(imageData, 0, 0)
  })

  return bytes;
}
