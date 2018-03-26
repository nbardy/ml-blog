import * as dl from 'deeplearn'

window.dl = dl;
function newGame(rows,cols)  {
  const size = rows * cols;
  const cells = new Int8Array(size);

  const buffer = dl.buffer([size],'int32',cells);
  return buffer;

}

function isEmpty(game, i) {
    return game.get(i) === 0;
}

function getMoveRandom(game) {
  var randSpot = Math.floor(game.size * Math.random())

  if(isEmpty(game,randSpot)) {
    return randSpot;
  } else {
    return getFirstEmptySquare(game);
  }
}

function getFirstEmptySquare(game) {
  for(let i = 0; i < game.size; i++) {

    if(game.get(i) === 0) {
      return i;
    }
  }
}

function getMoveClostestToICenter(game,opts) {
  const {COL_COUNT,ROW_COUNT} = opts;
  const middle = rowAndColToI([ROW_COUNT/2,COL_COUNT/2],opts)
  // const middle = 5500;
  var currentOffset = 0;

  while(!isEmpty(game,middle + currentOffset)) {
    // When negative flip the sign
    if(currentOffset / Math.abs(currentOffset) === 1) {
      currentOffset *= -1;
    }
    else {
      currentOffset = currentOffset * -1 + 1;
    }
    // When positive increment
  }

  return middle + currentOffset;
}


function rowAndColToI([row,col],{COL_COUNT}) {
  return (row * COL_COUNT) + col
}

function ItoRowAndCol(i,{ROW_COUNT, COL_COUNT}) {
  return [(i / COL_COUNT) >> 0, i % ROW_COUNT];
}


function neighbors(i, opts) {
  const {ROW_COUNT, COL_COUNT} = opts;;
  const [row, col] = ItoRowAndCol(i, opts)
  return [
    [row + 1 , col],
    [row - 1 , col],
    [row , col + 1],
    [row , col - 1]
  ].filter(function([row,col]) {
    return (row >= 0 && row < ROW_COUNT) && (col >= 0 && col < COL_COUNT)
  })
  .map(function(cell) {  return rowAndColToI(cell,opts); })
}

function addCellFreq(freqs, i, player) {
  // Add default values
  if(!freqs.get(i)) {
    freqs.set(i, new Map([["total", 0]]))
  }

  if(!freqs.get(i).get(player)) {
    freqs.get(i).set(player,0);
  }

  freqs.get(i).set(player,freqs.get(i).get(player) + 1)
  freqs.get(i).set("total",freqs.get(i).get("total") + 1)
}

// A map of indexs containing freq of each players neighbor
// e.g. {4:  {-1: 2, 1: 1},
//       12: {-1: 1, 1: 3}}
// In this example:
//   'square 4' has two neighbors of 'player -1' and 1 neighbors of player 1
//   'square 12 has one neighbors of 'player -1' and 3 neighbors of player 1
//
function neighborFrequencies(game,opts) {
  const freqs = new Map();

  for(let i = 0; i < game.size; i++) {
    if(game.get(i) === 0) {
      neighbors(i,opts).forEach(
        function(neighbor) {
          const neighborValue = game.get(neighbor);
          if(neighborValue !== 0)  {
            addCellFreq(freqs,i,neighborValue);
          }
        })
    }
  }

  return freqs;
}

// Spreads plague to empty squares
function progressBoard(game, opts) {
  const {COL_COUNT,ROW_COUNT} = opts;
  const shape = [ROW_COUNT,COL_COUNT];

  var total;
  var accumlator;
  var newVal;
  // const neighborFreqs = neighborFrequencies(game,opts);

  const next = dl.tidy("spread", () => {
    const init = game.toTensor();
    const x = dl.tensor1d([1, 2, 3, 4]);

    const init2d = init.reshape(shape);
    // TODO: Replace slice and stack with gather
    // init2d.print();
    const padded    = init2d.pad([[1,1],[1,1]]);
    const up        = padded.slice([0,1],shape);
    const left      = padded.slice([2,1],shape);
    const down      = padded.slice([1,0],shape);
    const right     = padded.slice([1,2],shape);
    const neighbors = dl.stack([up,down,left,right],2);
    const random    = dl.randomUniform(neighbors.shape);
    const weights   = random.mul(dl.cast(neighbors, 'float32'));
    const prob      = weights.sum(2);
    const rounded   = dl.cast(prob.mul(dl.scalar(2)), 'int32');
    // const probInc   = dl.mul(rounded,dl.scalar(50)); // To make Effect less random
    const nextVals  = rounded.clipByValue(-1,1);
    const isZero    = init2d.notEqual(dl.scalar(0,'int32'))
    const final     = dl.where(isZero,init2d,nextVals)

    return final;
  });


  return next.as1D().buffer();
}

function gameEnded(game) {
  for(let i = 0; i < game.size; i++) {

    if(game.get(i) === 0) {
      return false;
    }
  }

  return true;
}


export {rowAndColToI, getMoveRandom, getMoveClostestToICenter, gameEnded, newGame, progressBoard}
