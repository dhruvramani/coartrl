# Coarticulation Tensorflow

Run the following commands on my [clone](https://github.com/dhruvramani/transition) to generate the trained primitive policies. 

```
python3 -m rl.main --prefix toss_ICLR2019 --env JacoToss-v1 --hrl False --num_rollouts 10000
python3 -m rl.main --prefix hit_ICLR2019 --env JacoHit-v1 --hrl False --num_rollouts 10000
```

All the policies will be saved as pickle files in `./policies`. To train and save the meta policy, run the following command on the clone repo 

```
python3 -m rl.main
```
Save the policies folder from that repo to this. 
The `main.py` here contains the main coarticulation functions. It loads the primitive and meta policies, determines the order in which it will execute the primitive policies (for large number, currently only 2 so remember lol).

The reward calculation (as per the coarticulation algorithm) is done in `rollouts.py` in the function `traj_segment_generator_coart` and the new policy is trained using the normal `RLTrainer` in `trainer_rl.py`.

To run the code in this, 
```
cd transition
python3 main.py
```