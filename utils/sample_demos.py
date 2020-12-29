# def sample_demos(env, policy, n_episode=20, epi_length=10):
#     obs = env.reset()
#     for _ in range(n_episode):
#         for i in range(epi_length):
#             action = policy(obs)
#             yield [obs, action]
#             obs, _ = env.step(action)
#         obs = env.reset()

def sample_demos(env, policy, n_episode=1, epi_length=29):
    obs = env.reset()
    trajectory = convert_trajectory(policy)
    for _ in range(n_episode):
        for i in range(epi_length):
            # policy = trajectory dictionary
            # trajectory[i] = [obs, action]
            yield trajectory[i]
        obs = env.reset()

def convert_trajectory(traj_dict):
    # traj_dict = {"0": {"tile": 22, "heading": 0, "action": "go_forth"}, "1": {"tile": 22, "heading": 90, "action": "turn_left"}, "2": {"tile": 19, "heading": 90, "action": "go_forth"}, "3": {"tile": 13, "heading": 90, "action": "go_forth"}, "4": {"tile": 13, "heading": 0, "action": "turn_right"}, "5": {"tile": 14, "heading": 0, "action": "go_forth"}, "6": {"tile": 15, "heading": 0, "action": "go_forth"}, "7": {"tile": 16, "heading": 0, "action": "go_forth"}, "8": {"tile": 16, "heading": 270, "action": "turn_right"}, "9": {"tile": 20, "heading": 270, "action": "go_forth"}, "10": {"tile": 24, "heading": 270, "action": "go_forth"}, "11": {"tile": 24, "heading": 180, "action": "turn_right"}, "12": {"tile": 23, "heading": 180, "action": "go_forth"}, "13": {"tile": 23, "heading": 270, "action": "turn_left"}, "14": {"tile": 23, "heading": 0, "action": "turn_left"}, "15": {"tile": 24, "heading": 0, "action": "go_forth"}, "16": {"tile": 24, "heading": 90, "action": "turn_left"}, "17": {"tile": 20, "heading": 90, "action": "go_forth"}, "18": {"tile": 16, "heading": 90, "action": "go_forth"}, "19": {"tile": 11, "heading": 90, "action": "go_forth"}, "20": {"tile": 8, "heading": 90, "action": "go_forth"}, "21": {"tile": 8, "heading": 180, "action": "turn_left"}, "22": {"tile": 7, "heading": 180, "action": "go_forth"}, "23": {"tile": 7, "heading": 90, "action": "turn_right"}, "24": {"tile": 2, "heading": 90, "action": "go_forth"}, "25": {"tile": 2, "heading": 0, "action": "turn_right"}, "26": {"tile": 3, "heading": 0, "action": "go_forth"}, "27": {"tile": 4, "heading": 0, "action": "go_forth"}
    traj_list = []
    for i in range(len(traj_dict)):
        step = str(i)
        print("tile:", traj_dict[step]["tile"])
        state = traj_dict[step]["tile"]*4
        if traj_dict[step]["heading"] == 90:
            state += 1
        elif traj_dict[step]["heading"] == 180:
            state += 2
        elif traj_dict[step]["heading"] == 270:
            state += 3
        if traj_dict[step]["action"] == "go_forth":
            action = 0
        elif traj_dict[step]["action"] == "turn_left":
            action = 1
        elif traj_dict[step]["action"] == "turn_right":
            action = 2
        print('state:', state)
        traj_list.append([state, action])
    return traj_list
