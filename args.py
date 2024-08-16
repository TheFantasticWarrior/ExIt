import pathlib


path = f"{pathlib.Path().absolute()}/"
debug = 0
load = 1
reset_dataset=1
render = 1
gpu = True # not debug
#ob_space = (1476,)
action_space = 10

seed = 4000
nenvs = 8 if debug else 128
ntrees = nenvs

total_timesteps = 1e9

samplesperbatch = 8 if debug else nenvs
tree_iterations = 10 if debug else int(10000)
max_record = int(1e5)

loops = int(total_timesteps // samplesperbatch // tree_iterations)+1

lose_rew=-25
