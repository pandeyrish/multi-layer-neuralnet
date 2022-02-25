# multi-layer-neural-net

The project is used to build and train a simple multi-layer neural network using Keras and TensorFlow. The network is used to classify collisions based on 1D particle interactions into one of four classes:
• NO_COLLISION   Particles continue to travel in their respective directions.
• INELASTIC Both particles merge after collision and continue traveling as one in one direction.
• ELASTIC Particles collide and travel in new directions.
• INELASTIC_DECAY   Particles collide, travel as one in same direction for some time and then split into two and travel in new directions.


Commands to execute:

python mini_geant.py 1000 1 5 1 5 1 5 folder_name 

python interact_net.py 5fold folder_name model.h5 

python interact_net.py train folder_name model.h5 

python interact_net.py test folder_name model.h5 
