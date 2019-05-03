# Transition based Coarticulation
## Plan 

Train the primitive policies using the code in the transition repo's and save them as pickle files (not able to do it in my machine). 

Load up the primitive policies after creating an object and make the algo as decided on the paper. Will use fairly simple-written code as compared to the transition repo. Will have to look over at the environment code.

Can use the meta-policy from the transition paper to predict the order of the primative policies. Use `traj_segment_generator`` function in `rollouts.py`.
