import * as tf from '@tensorflow/tfjs';

tf.Tensor.prototype.stackNTimes = function(n) { 
  return this.expandDims(0).tile([n,1])
}

// Conerts a tensor of [[x,y], ...] to [i,...] coords
// where i = y * width + x
//
// Returns a tensor of type 'int32'
tf.Tensor.prototype.xytoI = function(width) {
  const rep = tf.tensor1d([1, width], 'int32').stackNTimes(this.size / 2)
  const indices = tf.round(this).toInt().mul(rep).sum(1);

  return indices;
}

//  Takes a tensor and returns a sum of the squares.
tf.Tensor.prototype.magnitude = function(axis) {
  return this.square().sum(axis || 1)
}

//  Takes a tensor and returns a sum of the squares.
tf.Tensor.prototype.magnitude = function(axis) {
  return this.square().sum(axis || 1)
}
