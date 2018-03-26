import {newGame,gameEnded,getMoveRandom,getMoveClostestToICenter,progressBoard} from './data.js';
import {newCanvas,drawGame} from './draw.js';
import {runml} from './nn.js';
import * as dl from 'deeplearn'

const ROW_COUNT  = 200,
      COL_COUNT  = 200,
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
  const canvas = newCanvas(COL_COUNT, ROW_COUNT, CELL_WIDTH)
  var game = newGame(ROW_COUNT, COL_COUNT);

  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)

  const opts = {
    ROW_COUNT: ROW_COUNT,
    COL_COUNT: COL_COUNT,
    CELL_WIDTH: CELL_WIDTH,
    canvas: canvas,
    board: board
  }

  drawGame(game, opts)
  // dl.ENV.set('DEBUG',true)

  var player = 1;
  var move;
  const next = function() {
    move = getMoveRandom(game,opts);
    game.set(player,move);
    player = -1

    drawGame(game, opts);

    move = getMoveRandom(game,opts);
    game.set(player,move);
    player = 1


    game = progressBoard(game,opts);

    if (!gameEnded(game)) {
      gameThreads.push(setTimeout(next, TURN_SPEED));
    } else {
      drawGame(game, opts);
    }
  }

  next()
}

function clean() {
  document.body.innerHTML = ""
  gameThreads.forEach(v => clearTimeout(v))
  gameThreads = [];
}

runml()
