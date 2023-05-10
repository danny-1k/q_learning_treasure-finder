import os
import sys
from time import sleep
import numpy as np

refresh_sleep_time = .03

num_states = 8
num_actions = 2  # left, right
num_episodes = 20

# states => . . . . . T

states = "."*(num_states-1)+"T"
q_table = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = .9
epsilon = .6

rewards = {
    ".": -1,  # negative reward for landing in "."
    "T": 1
}

clear_command = 'clear' if os.name == 'posix' else 'CLS'


def print_position(current_index):
    os.system(clear_command)
    states_ = list(states)
    states_[current_index] = "+"
    states_ = "".join(states_)
    sys.stdout.write(states_)
    sys.stdout.flush()

    sleep(refresh_sleep_time)


def update_q_table(state, action, reward, q_pred, q_target):
    q_table[state, action] += alpha * (reward + gamma*(q_target - q_pred))


def sample_action(state):
    if np.random.rand() < epsilon:
        action = np.random.randint(2)  # left, right

    else:
        action = q_table[state].argmax()

    return action


def get_environment_feedback(state, action):
    if action == 1:  # right
        state += 1
    else:  # left
        state -= 1

    if state < 0:  # agent has moved to the left too much
        state = 0

    reward = rewards[states[state]]

    return state, reward


def main():

    for episode in range(num_episodes):
        current_state_index = 0

        print(f"\nEPISODE: {episode}")
        sleep(1)

        while current_state_index != (num_states-1):
            print_position(current_index=current_state_index)
            action = sample_action(current_state_index)

            current_state_index, reward = get_environment_feedback(
                state=current_state_index,
                action=action
            )

            q_pred = q_table[current_state_index, action]

            if current_state_index < (num_states-1):
                q_target = q_table[current_state_index + 1].max()
            else:
                q_target = reward

            update_q_table(
                state=current_state_index,
                action=action,
                reward=reward, 
                q_pred=q_pred, 
                q_target=q_target
            )

        print_position(current_index=current_state_index)


if __name__ == "__main__":

    main()

    print("Q_VALUES")
    print(q_table)
