from jasonpong.envs.pong import Pong


def play_game():
    env = Pong()
    obs = env.reset()
    is_game_over = False
    players = list(range(2))
    while not is_game_over:
        # state = env.get_state()
        # env.step(1)
        for player in players:
            obs, reward, is_game_over, _ = env.step(1)
            env.render()


if __name__ == '__main__':
    play_game()
