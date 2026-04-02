# StarTrace

Creates initial conditions for N-body simulations consisting of N particles
in N\_sc subclusters, each in a plummer distribution. The user can set the 
virial parameter of the subclusters and the system as a whole. The subclusters
are also set up with a coherent velocity towards the system center of mass.

Coming soon... A graph neural network for recovering initial substructure in star clusters. Including a library to setup and execute N-body simulations of varying degrees of sub-clustering for training/testing.

### Usage

see generate\_ics.py for example

### Running petar

Install petar. generate initial conditions from the outputted text file with:

petar.init ics.txt

Then run petar with:

petar -u 1 -r 0.1 ics.txt.input
>>>>>>> 7d5cf48 (first commit)
