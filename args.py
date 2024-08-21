import pathlib


path = f"{pathlib.Path().absolute()}/save"
debug = 0
load = 1
reset_dataset=0
render = 0
gpu = True # not debug
#ob_space = (1476,)
action_space = 10

seed = 5
nenvs = 8 if debug else 16
ntrees = 16

total_timesteps = 1e9

samplesperbatch = 8 if debug else nenvs
init_iterations=1000
iterations = 10 if debug else int(100000)
max_record = int(1500)

loops = 10

lose_rew=-2.5
