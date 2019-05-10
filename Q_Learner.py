from random import randint,uniform
import csv

try:
    xrange
except:
    xrange = range
    

class Q_Learner:
    def __init__(self, exploration_rate):
        self.episolon = exploration_rate
        self.alpha = 0.9

        self.gamma = 0.8
        self.q_table = {}
        self.action_sequence = {}
        self.state_action= {}

    def build_state_action_key(self,state,action):
        return tuple((tuple(state),action))

    def get_q_val(self, state, action_id):
        state_action_key = self.build_state_action_key(state,action_id)
        if state_action_key not in self.q_table:
            return 0, 0

        return self.q_table[state_action_key]

    def get_best_q_val(self, state, num_actions):
        best_q_t1_val = None
        for action_id in xrange(num_actions):
            state_action_key = self.build_state_action_key(state,action_id)
            if state_action_key not in self.q_table:
                v = 0
            else:
                v, attempts = self.q_table[state_action_key]

            if best_q_t1_val is None or v>best_q_t1_val:
                best_q_t1_val = v

        return best_q_t1_val

    def dump_qvals(self):
        with open('qvals.csv','w') as f:
            fcsv = csv.writer(f,lineterminator='\n')
            fcsv.writerow(['State','Values','Attempts'])

            for state_action_key in self.q_table:
                fcsv.writerow([state_action_key, self.q_table[state_action_key][0], self.q_table[state_action_key][1]])

    def select(self, state, num_actions, learn, actions):
        best_action_id = None
        q_val = None
        best_q_t1_val = None
        self.action_sequence[tuple(state)] = num_actions
        self.state_action[tuple(state)] = tuple(actions)

        if uniform(0,1)<self.episolon and learn:
            #Choose random action
            best_action_id = randint(0,num_actions-1)
            q_val, attempts = self.get_q_val(state,best_action_id)
            best_q_t1_val = 0
        else:
            for action_id in xrange(num_actions):
                next_q_val, attempts = self.get_q_val(state,action_id)
                if learn:
                    next_q_val += uniform(1e-6,1e-8)
                if best_q_t1_val is None or next_q_val>best_q_t1_val:
                    best_q_t1_val = next_q_val
                    best_action_id = action_id
        return best_action_id

    def update(self, state_list, action_list, final_state, reward):

        prev_reward = reward

        for i in xrange(len(state_list)-1,-1,-1):
            prev_state = state_list[i]            
            prev_action = action_list[i]
            specific_action = final_state[i][prev_action]

            q_val, attempts = self.get_q_val(prev_state, prev_action)
            
            best_state_q_val = prev_reward

            if i == len(state_list)-1:
                new_q = best_state_q_val + attempts*q_val
                attempts += 1
                new_q /= float(attempts)
            else:
                attempts = 1
                new_q = (1 - self.alpha) * q_val + self.alpha * (self.gamma * best_state_q_val)

            prev_state_action_key = self.build_state_action_key(prev_state, prev_action)
            self.q_table[prev_state_action_key] = (new_q,attempts)
            state       = prev_state
            num_actions = self.action_sequence[tuple(state)]
            prev_reward = self.get_best_q_val(state, num_actions)

        self.action_sequence = {}