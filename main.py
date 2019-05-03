from pong import Pong

def play_game():
    env = Pong()

    while not env.is_gameover():
        state = env.get_state()
        print ('Cyclce:{} Player:{} Bat:{} Ball_P:({},{}) Ball_V:({},{})'.format(env.cycle,env.player_turn,state[0],state[1],state[2],state[3],state[4]))
        env.execute(1)

play_game()