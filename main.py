from jasonpong.envs.pong import Pong


def play_game():
    env = Pong()
    initial_obs = env.reset()
    obs = [initial_obs, initial_obs]
    is_game_over = False
    players = list(range(2))
    while not is_game_over:
        for player in players:
            if obs[player][0] < 5:
                action = 2  # right
            elif obs[player][0] > 5:
                action = 1  # left
            else:
                action = 0
            next_obs, reward, is_game_over, _ = env.step(action)
            obs[player] = next_obs
            env.render()
            if reward:
                print('winner:', player)


if __name__ == '__main__':
    play_game()
