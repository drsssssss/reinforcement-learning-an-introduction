import numpy as np

GOAL = 100
STATES = np.arange(GOAL + 1)
HEAD_PROB = 0.4


def compute_state_value():
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    iteration = 0
    while True:
        old_state_value = state_value.copy()
        for state in STATES[1:GOAL]:
            action_returns = []
            actions = np.arange(min(state, GOAL - state) + 1)
            for action in actions:
                win = state + action
                lose = state - action
                action_returns.append(HEAD_PROB * state_value[win] + (1 - HEAD_PROB) * state_value[lose])
            state_value[state] = np.mean(action_returns)

        delta = abs(state_value - old_state_value).max()
        iteration += 1
        if delta < 1e-9:
            break

    return iteration, state_value


if __name__ == '__main__':
    iteration, state_value = compute_state_value()
    print(f"Number of iterations for policy evaluation to converge: {iteration}")
