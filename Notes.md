I want to have one big matrix

NxN which represents forces over an area.
A force is [xForce, yForce].
Magnitude is 0-1

0.1 

Questions to answer
-------------------
How will I store the force field matrix?
How will I train the force field matrix?
How will I store the partciles?
How will I update the particles?


How will I store. 


High Level Algo

Add a particle to the particle list, Give it a spawn time, position, velocity. 
Constantly update particles based on force field.
Update 
    - Adds a bit to the score based on position
    - Changes position 
    - Records the index of the force arrow effecting it.

Train
    - Takes the result of the pass.
    - Give it a score. 0 is perfect. Higher is most distance
    - Distribute error across all nodes previously hit.  

Remove particles that leave screen.
When I reach a certain amount of removed particles. 

Input => Matrix => Output

Input => Matrix => Output

.


