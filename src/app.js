import {newGame,gameEnded,getMove,progressBoard} from './data.js';
import {newCanvas,drawGame} from './draw.js';

const ROW_COUNT  = 300,
      COL_COUNT  = 300,
      CELL_WIDTH = 2,
      TURN_SPEED = 0;

var gameThreads = [];

if(module.hot) {
  console.log(module.hot);
  clean()
  module.hot.accept();
  start()
}

function start() {
  const opts = {
    ROW_COUNT: ROW_COUNT,
    COL_COUNT: COL_COUNT,
    CELL_WIDTH: CELL_WIDTH
  }

  const canvas = newCanvas(COL_COUNT*CELL_WIDTH, ROW_COUNT*CELL_WIDTH)
  const game = newGame(ROW_COUNT, COL_COUNT);

  document.body.appendChild(canvas)
  drawGame( canvas, game, opts)

  var player = 1;
  var move;

  const next = function() {
    // for (let i = 0 ; i < 10; i++) {
    move = getMove(game);
    game.set(player,move);
    player = -1

    // drawGame(canvas, game, opts);

    move = getMove(game);
    game.set(player,move);
    player = 1

    // drawGame(canvas, game, opts);

    progressBoard(game,opts);
    // drawGame(canvas, game, opts);

    if (!gameEnded(game)) {
      gameThreads.push(setTimeout(next, TURN_SPEED));
    } else {
      drawGame(canvas, game, opts);
    }
  }

  next()
}

function clean() {
  document.body.innerHTML = ""
  gameThreads.forEach(v => clearTimeout(v))
  gameThreads = [];
}
