# Coarticulation Tensorflow

Make sure the directories (this and my [clone](https://github.com/dhruvramani/transition)) are saved in this structure

```
|_ coartl
|_ transition

```

Run the following commands on my [clone](https://github.com/dhruvramani/transition) to generate the trained primitive policies. 

```
python3 -m rl.main --prefix toss_ICLR2019 --env JacoToss-v1 --hrl False --num_rollouts 10000
python3 -m rl.main --prefix hit_ICLR2019 --env JacoHit-v1 --hrl False --num_rollouts 10000
python3 -m rl.main --prefix toss_ICLR2019 --env JacoToss-v1 --hrl False --num_rollouts 10000 --is_collect_state True --num_evaluation_run 1000 # Jaco Toss env
python3 -m rl.main --prefix hit_ICLR2019 --env JacoHit-v1 --hrl False --num_rollouts 10000 --is_collect_state True --num_evaluation_run 1000 # Jaco Hit env
```

All the policies will be saved as pickle files in `./policies`. To train and save the meta policy, run the following command on the clone. 

```
python3 -m rl.main
```
This saves the checkpoints of all the policies required by us in `../../log`.
The `main.py` here contains the main coarticulation functions. It loads the primitive and meta policies, determines the order in which it will execute the primitive policies.

The reward calculation (as per the coarticulation algorithm) is done in `rollouts.py` in the function `traj_segment_generator_coart` and the new policy is trained using the normal `RLTrainer` in `trainer_rl.py`.

To run the code in this, 
```
cd transition
python3 main.py
```