#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution

    def update_distribution(distribution, observation):
        if observation is None: return distribution
        updated_distribution = robot.Distribution({state: prob * observation_model(state)[observation] for state, prob in distribution.items() if prob and observation_model(state)[observation]})
        updated_distribution.renormalize()
        return updated_distribution

    for n in range(num_time_steps - 1):
        forward_messages[n+1] = robot.Distribution()
        updated_distribution = update_distribution(forward_messages[n], observations[n])            
        for current_state, updated_prob in updated_distribution.items():
            for next_state, transition_prob in transition_model(current_state).items():
                forward_messages[n+1][next_state] += transition_prob * updated_prob

    backward_messages = [None] * num_time_steps
    backward_messages[-1] = robot.Distribution({state: 1 for state in all_possible_hidden_states})
    backward_messages[-1].renormalize()

    reverse_transition_model = {state: robot.Distribution() for state in all_possible_hidden_states}
    for previous_state in all_possible_hidden_states:
        for state, transition_prob in transition_model(previous_state).items():
            reverse_transition_model[state][previous_state] = transition_prob
    #for distribution in reverse_transition_model.values(): distribution.renormalize()
            
    for n in reversed(range(num_time_steps - 1)):
        backward_messages[n] = robot.Distribution()
        updated_distribution = update_distribution(backward_messages[n+1], observations[n+1])
        for current_state, updated_prob in updated_distribution.items():
            for previous_state, reverse_transition_prob in reverse_transition_model[current_state].items():
                backward_messages[n][previous_state] += reverse_transition_prob * updated_prob            

    marginals = [None] * num_time_steps # remove this
    for n in range(num_time_steps):
        marginal_distribution = robot.Distribution()
        for state in all_possible_hidden_states: #forward_messages[n].keys() + backward_messages[n].keys():
            marginal_distribution[state] = forward_messages[n][state] * backward_messages[n][state]
        marginals[n] = update_distribution(marginal_distribution, observations[n])

    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #



    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this
    
    def update_distribution(distribution, observation):
        if observation is None: return robot.Distribution({state: 1 for state in distribution.keys()})
        return robot.Distribution({state: observation_model(state)[observation] for state in distribution.keys() if observation_model(state)[observation]})
    
    messages = []
    messages.append(robot.Distribution({state: careful_log(prob) for state, prob in prior_distribution.items()}))
    arguments = [{}]
    possible_min_arg = ()
    for n in range(num_time_steps - 1):
        message = robot.Distribution()
        min_argument = {}
        for next_state in all_possible_hidden_states:
            possible_min = np.inf
            for current_state, updated_prob in update_distribution(messages[n], observations[n]).items():
                minus_log_prob = messages[n][current_state] - careful_log(transition_model(current_state)[next_state]) - careful_log(updated_prob)
                if minus_log_prob < possible_min:
                    possible_min = minus_log_prob
                    possible_min_arg = current_state
            
        
        # updated_distribution = update_distribution(messages[n], observations[n])            
        # for current_state, updated_prob in updated_distribution.items():
        #     possible_min = 3
        #     for next_state, transition_prob in transition_model(current_state).items():
        #         minus_log_prob = messages[n][next_state] - careful_log(transition_prob) - careful_log(updated_prob)
        #         if minus_log_prob < message[next_state]:
        #             message[next_state] = minus_log_prob
        #             min_arguments[next_state] = next_state
                    
                    
            if possible_min < np.inf:
                message[next_state] = possible_min
                min_argument[next_state] = possible_min_arg
        messages.append(message)
        arguments.append(min_argument)
        
    possible_min = np.inf
    possible_min_arg = ()
    for last_state, updated_prob in update_distribution(messages[-1], observations[-1]).items():
        minus_log_prob = messages[-1][last_state] - careful_log(updated_prob)
        if minus_log_prob < possible_min:
            possible_min = minus_log_prob
            possible_min_arg = last_state

    estimated_hidden_states[-1] = possible_min_arg

    for n in reversed(range(num_time_steps - 1)):
        estimated_hidden_states[n] = arguments[n+1][estimated_hidden_states[n+1]]

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this
    
    def update_distribution(distribution, observation):
        if observation is None: return robot.Distribution({state: 1 for state in distribution.keys()})
        updated_distribution = robot.Distribution({state: observation_model(state)[observation] for state in distribution.keys() if observation_model(state)[observation]})
        # updated_distribution.renormalize()
        return updated_distribution
    messages = []
    second_messages = []
    messages.append(robot.Distribution({state: careful_log(prob) for state, prob in prior_distribution.items()}))
    second_messages.append(robot.Distribution())
    arguments = [{}]
    second_arguments = [{}]
    possible_min_arg = ()
    possible_best_of_second_arg = ()
    possible_second_of_best_arg = ()
    for n in range(num_time_steps - 1):
        message = robot.Distribution()
        second_message = robot.Distribution()
        min_argument = {}
        second_argument = {}
        for next_state in all_possible_hidden_states:
            possible_min = np.inf
            second_of_best = np.inf
            for current_state, updated_prob in update_distribution(messages[n], observations[n]).items():
                best_minus_log_prob = messages[n][current_state] - careful_log(transition_model(current_state)[next_state]) - careful_log(updated_prob)
                if best_minus_log_prob < second_of_best:
                    if best_minus_log_prob >= possible_min:
                        second_of_best = best_minus_log_prob
                        possible_second_of_best_arg = current_state
                    else:
                        second_of_best = possible_min
                        possible_second_of_best_arg = possible_min_arg
                        possible_min = best_minus_log_prob
                        possible_min_arg = current_state

            best_of_second = np.inf
            for current_state, updated_prob in update_distribution(second_messages[n], observations[n]).items():
                second_minus_log_prob = second_messages[n][current_state] - careful_log(transition_model(current_state)[next_state]) - careful_log(updated_prob)
                if second_minus_log_prob < best_of_second:
                    best_of_second = second_minus_log_prob
                    possible_best_of_second_arg = current_state
            if possible_min < np.inf:
                if n == 56: print(possible_min, second_of_best, best_of_second)
                message[next_state] = possible_min
                min_argument[next_state] = possible_min_arg
            if second_of_best < best_of_second:
                second_message[next_state] = second_of_best
                second_argument[next_state] = possible_second_of_best_arg
                state = possible_second_of_best_arg
                previous_state = ()
                for t in range(1, n+1):
                    previous_state = arguments[-t][state]
                    second_messages[-t][state] = messages[-t][state]
                    second_arguments[-t][state] = previous_state
                    state = previous_state
            elif best_of_second < np.inf:
                second_message[next_state] = best_of_second
                second_argument[next_state] = possible_best_of_second_arg

        messages.append(message)
        arguments.append(min_argument)
        second_messages.append(second_message)
        second_arguments.append(second_argument)
        if n == 54 or n == 55: print(arguments[1], second_arguments[1])


        
    possible_min = np.inf
    possible_min_arg = ()
    second_of_best = np.inf
    second_of_best_arg = ()
    for last_state, updated_prob in update_distribution(messages[-1], observations[-1]).items():
        minus_log_prob = messages[-1][last_state] - careful_log(updated_prob)
        if minus_log_prob < second_of_best:
            if minus_log_prob < possible_min:
                second_of_best = possible_min
                second_of_best_arg = possible_min_arg
                possible_min = minus_log_prob
                possible_min_arg = last_state
            else:
                second_of_best = minus_log_prob
                second_of_best_arg = last_state
    
    best_of_second = np.inf
    best_of_second_arg = ()
    for last_state, updated_prob in update_distribution(second_messages[-1], observations[-1]).items():
        minus_log_prob = second_messages[-1][last_state] - careful_log(updated_prob)
        if minus_log_prob < best_of_second:
            best_of_second = minus_log_prob
            best_of_second_arg = last_state

    if second_of_best < best_of_second:
        estimated_hidden_states[-1] = second_of_best_arg
        second_arguments = arguments
    else: estimated_hidden_states[-1] = best_of_second_arg

    for n in reversed(range(num_time_steps - 1)):
        estimated_hidden_states[n] = second_arguments[n+1][estimated_hidden_states[n+1]]

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 10
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
