import {Array1D, ENV, Scalar, NDArrayMathGPU, Graph} from 'deeplearn'

async function foo() {
  const math = ENV.math;
  const a = Array1D.new([1, 2, 3]);
  const b = Scalar.new(2);

  const result = math.add(a, b);

  var t = await result.data();  // Float32Array([3, 4, 5])
  return t;  // Float32Array([3, 4, 5])
}

function newGame(rows,cols)  {
  const cells = [];

  for(let i = 0; i < rows * cols; i++) {
    cells.push(0);
  }

  return Array1D.new(cells);
}

function getMove(game, player) {
  var rand = Math.floor(game.size * Math.random())
  if(game.get(rand) === 0) {
    return rand
  }

  for(let i = 0; i < game.size; i++) {

    if(game.get(i) === 0) {
      return i;
    }
  }
}

function rowAndColToI([row,col],{COL_COUNT}) {
  // console.log("toRow", [row,col], "->", (row * COL_COUNT) + col)
  return (row * COL_COUNT) + col
}

function ItoRowAndCol(i,{ROW_COUNT, COL_COUNT}) {
  // console.log("toI", i, "->", [(i / COL_COUNT) >> 0, i % ROW_COUNT])
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
  var total;
  var accumlator;
  var newVal;
  const neighborFreqs = neighborFrequencies(game,opts);

  neighborFreqs.forEach(
    function(freqMap,i) {
      accumlator = 0;
      total = freqMap.get("total")
      freqMap.delete("total")
      const pairs = Array.from(freqMap);
      const chance = Math.random();
      pairs.forEach(
        function ([k,v]) {
          accumlator += 1.0 * v / total


          if(chance < accumlator) {
            game.set(k,i)
          }
        })
    })

  return game;
}

function gameEnded(game) {
  for(let i = 0; i < game.size; i++) {

    if(game.get(i) === 0) {
      return false;
    }
  }

  return true;
}

export {rowAndColToI, getMove, gameEnded, newGame, progressBoard}
