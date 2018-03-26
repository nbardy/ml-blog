import {rowAndColToI} from './data'

function newCanvas(w,h,scale) {
  var canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h
  canvas.setAttribute("style", "width: " + w*scale + "px;");
  canvas.setAttribute("style", "width: " + h*scale + "px;");

  return canvas
}

function updateIn(scores, k, fn, init_v) {
  const v = scores.get(k) || init_v;
  scores.set(k,fn(v));
  return scores;
}

const colors = {
  "-1": "red",
  "0":  "grey",
  "1":  "green"
};

const names = {
  "-1": "getClose",
  "0":  "empty",
  "1":  "Random"
};

function drawGame(game, opts) {
  const {CELL_WIDTH, ROW_COUNT, COL_COUNT, board, canvas} = opts;
  const ctx = canvas.getContext("2d");
  var imgData = ctx.createImageData(COL_COUNT, ROW_COUNT);
  var data = imgData.data;
  var scores = new Map();

  // Draw grid and calculate score;
  for(let i = 0; i < data.length; i++) {
    var color;
    const mark = game.get(i);

    color = colors[mark];
    data[i*4+3] = 255;
    if(mark == -1) {
      data[i*4 + 1] = 255;
    } else if(mark == 1) {
      data[i*4 + 2] = 255;
    } else {
      // Nothing
    }

    // scores = updateIn(scores,mark,function(v) {
    //   if(!(mark===0)) {
    //     return v + 1;
    //   }
    //   else {
    //     return v;
    //   }
    // },0);
  }


  // Draw Score
  var boardText = "";
  [1,-1].forEach(function(k) {

    boardText += (" " + colors[k] + " " + names[k] + ": " + scores.get(k) + "\n");
  })

  board.innerText = boardText;
  ctx.putImageData(imgData,0,0)

  return true;
}

function drawScore(ctx,game,opts){
  return false
}

export {newCanvas,drawGame}
