path = "/home/exit/save"
debug = False
load = True
gpu = not debug
ob_space = (1476,)
action_space = 10

seed = 1
render = True
nenvs = 16
ntrees = 7

total_timesteps = 1e9
samplesperbatch = 64 if debug else 256
tree_iterations = 1 if debug else int(1e5)
max_record = int(1e5)

loops = int(total_timesteps // samplesperbatch // tree_iterations)+1


# use dictionary and locals() to log because this looks simpler imo
losses_log = ["timestep", "re", "mr", "average", "avg_lines",
              "avg_atk", "lr", "ent_coef", "mean_p", "kpp"]
