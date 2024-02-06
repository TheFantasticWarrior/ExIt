path = "save"

ob_space = (1477,)
action_space = 10

seed = 1
render = 1
nenvs = 16

total_timesteps = 1e9
samplesperbatch = 250
max_record = 1e5

loops = int(total_timesteps // samplesperbatch // nenvs)+1


# use dictionary and locals() to log because this looks simpler imo
losses_log = ["timestep", "re", "mr", "average", "avg_lines",
              "avg_atk", "lr", "ent_coef", "mean_p", "kpp"]

debug = False
