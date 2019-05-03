import numpy as np
class Pong:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ball_pos = np.asarray([5,5])
        self.bat_pos = [0,0]
        self.ball_velocity = np.asarray([1,1])
        self.gameover = False
        self.player_turn = 0
        self.winner = None
        self.cycle = 0


    def update(self):
        self.cycle += 1
        self.ball_pos += self.ball_velocity
        # bounce off bat
        if (self.ball_pos[1] == 1 and abs(self.ball_pos[0]-self.bat_pos[0])<=1 ) or (self.ball_pos[1] == 10 and abs(self.ball_pos[0]-self.bat_pos[1])<=1 ): #player 1 bat y axis
            self.ball_velocity[1] *= -1
        # reflect off side walls
        if self.ball_pos[0] >= 10 or self.ball_pos[0] <= 0:
            self.ball_velocity[0] *= -1
        # lose and win conditions
        if self.ball_pos[1] >= 11:
            self.gameover = True
            self.winner = 0
        if self.ball_pos[1] <= 0:
            self.gameover = True
            self.winner = 1

    def render(self):
        pass

    def get_state(self):
        return np.asarray([self.bat_pos[self.player_turn]]+list(self.ball_pos)+list(self.ball_velocity))

    def execute(self, action_id):
        if self.gameover:
            return
        # 3 actions , left , right ,nothing
        if action_id == 0:
            self.bat_pos[self.player_turn ] = max(self.bat_pos[self.player_turn ]-1 , 0)
        if action_id == 1:
            self.bat_pos[self.player_turn ] = min(self.bat_pos[self.player_turn ]+1, 10)


        self.player_turn = (self.player_turn+1) % 2
        if self.player_turn == 0:
            self.update()


    def is_gameover(self):
        return self.gameover