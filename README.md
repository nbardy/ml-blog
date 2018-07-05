# Force Field Art

## Introduction 
Inspired by advances in machine learning training running on the GPU in a browser. I've become motivated to build art and demo applications of machine learning to explore the field and help pass on what I learn to others. This demo is a force field which moves particles around that is trained based on different evaluation particles of function position. The goal is 

# Current state

I tried running it with tensorflow's gradient calculation. And it was able to train a reasonable simple problem(Getting all particles to the middle). But it became unusably slow when exposed to denser force fields or trying to train over multiple frames. Currently I'm reasearching and implementing some Derivate Free optimization algorithms for training. Another approach I would like to explore is calculating swift gradient estimates.


# Development

Setup dependencies with
```
yarn install
```

To build live and interactive run 

```
yarn run start:dev
```

To build the dev build to `dist` run:
```
yarn run build:dev
```

There is currently no production build setup.

# Example Photo

![A photo showing a screenshot of particles moving around a force field and the relevant convergance chart.](resources/screenshot-july-4-2018.png)
