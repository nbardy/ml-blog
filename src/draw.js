import {rowAndColToI} from './data'

function newCanvas(w,h) {
  var canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h

  return canvas
}

function drawGame(canvas, game, opts) {
  const {CELL_WIDTH, ROW_COUNT, COL_COUNT} = opts;
  const ctx = canvas.getContext("2d");
  for(let r = 0; r < ROW_COUNT; r++) {
    for(let c = 0; c < COL_COUNT; c++) {

      var color;

      switch(game.get(rowAndColToI([r,c],opts))) {
        case 0:
          color = "grey";
          break;
        case 1:
          color = "red";
          break;
        case -1:
          color = "green";
          break;
      }
      ctx.fillStyle = color;
      ctx.fillRect(c*CELL_WIDTH,r*CELL_WIDTH, CELL_WIDTH, CELL_WIDTH);
    }
  }
}

export {newCanvas,drawGame}
