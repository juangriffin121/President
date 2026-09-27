from president.experiments.positional_dynamics.table_anim import animate_trial, save_animation

trial_id = "5_Linear"

anim = animate_trial(f"president/experiments/positional_dynamics/run{trial_id}.csv", trial_id=trial_id, game_range=(0,1000), debug=True)
save_animation(anim, f"president/experiments/positional_dynamics/run{trial_id}.animation.mp4", fps=4)
