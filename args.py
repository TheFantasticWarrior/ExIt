import pathlib


path = f"{pathlib.Path().absolute()}/"
debug = 0
load = 1
render = 1
gpu = True # not debug
#ob_space = (1476,)
action_space = 10

seed = 1
nenvs = 8 if debug else 128
ntrees = nenvs

total_timesteps = 1e9

samplesperbatch = 8 if debug else nenvs*2
tree_iterations = 10 if debug else int(10000)
max_record = int(1e5)

loops = int(total_timesteps // samplesperbatch // tree_iterations)+1


# use dictionary and locals() to log because this looks simpler imo
losses_log = ["timestep", "re", "mr", "average", "avg_lines",
              "avg_atk", "lr", "ent_coef", "mean_p", "kpp"]
