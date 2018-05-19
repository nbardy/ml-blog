import '~/helpers.js';
import * as tf from '@tensorflow/tfjs';


export function drawParticles(canvas, [posTensor, velTensor] , config) {
  const {width,height} = config;
  const bytes = new Uint8ClampedArray(width * height * 4); 

  tf.tidy(() => {
    // const positions  = posTensor.data();
    // const velocities = velTensor.data();

    const particleCount = posTensor.shape[0]

    // Transform to I index equivlanet for imageBuffer insertion
    const posIdx = posTensor.xytoI(width);
    // Transform to a magnitude value to be used as color
    const velMag =    velTensor.magnitude();

    for(let i = 0, imgIndex, color; i < posIdx.size; i++) {
      imgIndex = posIdx.get(i) * 4

      bytes[imgIndex] = 256
      bytes[imgIndex + 1] = 1;
      bytes[imgIndex + 3] = 255;
    }

    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(bytes, width, height)
    ctx.putImageData(imageData, 0, 0)
  })

  return bytes;
}
