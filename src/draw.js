import '~/helpers.js';
import * as tf from '@tensorflow/tfjs';
import img from '~/arrow.png';

export function drawParticles(imgData, [posTensor, velTensor] , config) {
  const {width,height} = config;
  const velScale = 2 * config.maxVel*config.maxVel

  const bytes = imgData.data;
  // const bytes = new Uint8ClampedArray(width * height * 4); 

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
      red      = colors.get(i,0)
      green      = colors.get(i,1)

      bytes[imgIndex] = 125 + 125 * red 
      bytes[imgIndex + 2] = 125 + 125 * green;
      bytes[imgIndex + 3] = 255;
    }

    const imageData = new ImageData(bytes, width, height)
  })

  return bytes;
}

export function drawRotated(ctx,image,x,y,width,height,r) {
  ctx.translate(x, y);
  ctx.rotate(r);
  ctx.drawImage(image, -width / 2, -height / 2, width, height);
  ctx.rotate(-r);
  ctx.translate(-x, -y);
}

var arrowImage = new Image();
arrowImage.src = img;

export function drawField(canvas, field, config) {
  const totalx = config.width * config.density;
  const totaly = config.height * config.density;

  const imgWidth = 1 / config.density;
  const imgHeight = 1 / config.density;

  var x, y, i, forcex, forcey, xmag, ymag, r;
  const ctx = canvas.getContext("2d");

  // ctx.translate(-imgHeight/2,imgWidth/2);

  for(var y = 0; y < totaly; y++) {
    // ctx.translate(imgHeight, 0);

    for(var x = 0; x < totalx; x++) {
      i = x + y * totalx;

      xmag = field.get(i,0),
      ymag = field.get(i,1);
      // TODO: Do the math here on GPU
      const mag = Math.min(0.4,Math.sqrt(xmag * xmag + ymag*ymag))
      r = Math.atan2(ymag, xmag);

      if(config.drawField) { 
        drawRotated(ctx, 
          arrowImage, 
          x * imgWidth + imgWidth/2, 
          y  * imgHeight + imgHeight/2, 
          imgWidth * mag, 
          imgHeight * mag,
          r)
      }
    }
  }
}

export function drawScene(canvas, particles, field, config) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if(config.drawField) {
    drawField(canvas, field, config);
  }

  const imgData = ctx.getImageData(0,0,canvas.width,canvas.height);

  drawParticles(imgData, particles, config);
  ctx.putImageData(imgData, 0, 0)
}
