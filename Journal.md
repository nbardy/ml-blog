# Journal
This is something new I'm trying. I end up writing tons of notes anyway when writing software might as well keep them all in source control

###~July 2, 2018

I tried running it with tensorflow's gradient calculation. And it was able to train a reasonable simple problem(Getting all particles to the middle). But it became unusably slow when exposed to denser force fields or trying to train over multiple frames. Currently I'm reasearching and implementing some Derivate Free optimization algorithms for training. Another approach I would like to explore is calculating swift gradient estimates.

###July 6th, 2018

Tweaked some variables and was able to get it converge to  a near zero loss. Very happy with this hoping it is relevatively consistent and I didn't just get lucky. This was a big realief. It proved two things:
    1) Derivate-Free optimizers can optimize the model. I was worried the problem might be too complex and the only approach would be a slow derivative based one.
    2) It showed I can write my own optimizer. It feels good to peel back the hood on tensorflow and make some of the magic happen myself. 
