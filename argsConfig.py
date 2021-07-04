# Import the argparse library
import argparse


def getParams():
    # Creating the parser
    argsParser = argparse.ArgumentParser(
        description='Reinforcement Learning by using a MDP/REINFORCE to solve CVRP')

    # Add arguments
    # ------------------
    # Convert
    argsParser.add_argument('--convert', default=False, action='store_true', help='entering convert mode')

    # Training
    argsParser.add_argument('--train', default=False, action='store_true', help='entering training mode')
    argsParser.add_argument('--num_episodes', default=115, type=int,
                            help="Define number of episodes for the training process")
    argsParser.add_argument('--max_steps', default=10000, type=int,
                            help="Define the number of maximal steps that can be taking for the training process")

    # Testing
    argsParser.add_argument('--test', default=False, action='store_true', help="entering test mode")

    # Agent
    argsParser.add_argument('--agent', default='policy_gradient_agent', help="Use the Policy Gradient Agent")

    argsParser.add_argument('--stay_duration', default=5.0, type=float,
                            help="Define stay duration")

    argsParser.add_argument('--learning_rate', default=2 ** -3, type=float,
                            help="Define learning rate")

    argsParser.add_argument('--discount_factor', default=0.95, type=float,
                            help="Define discount factor")

    argsParser.add_argument('--exploration_factor', default=0.05, type=float,
                            help="Define exploration factor")

    argsParser.add_argument('--increasing_factor', default=0.95, type=float,
                            help="Define increasing factor, for increasing the weight.")

    argsParser.add_argument('--increasing_factor_good_episode', default=0.9, type=float,
                            help="Define increasing factor, for increasing the weight based on a good episode.")

    argsParser.add_argument('--decreasing_factor', default=1.03, type=float,# davor 1.05
                            help="Define decreasing factor, for decreasing the weight.")

    argsParser.add_argument('--decreasing_factor_good_episode', default=1.06, type=float,
                            help="Define decreasing factor, for decreasing the weight based on a good episode.")

    argsParser.add_argument('--baseline_theta', default=0.00001, type=float,
                            help="Define small number to converged to in baseline calculation.")

    argsParser.add_argument('--local_search_threshold', default=0.5, type=float,
                            help="Define the threshold to apply local search.")

    argsParser.add_argument('--policy_reset_threshold', default=-1, type=float, # davor -7, davor -1
                            help="Define the threshold to reset policies.")

    # ACO
    argsParser.add_argument('--aco_alpha_factor', default=0.5, type=float,
                            help="Define discount alpha for aco")
    argsParser.add_argument('--aco_beta_factor', default=1.2, type=float,
                            help="Define discount beta for aco")
    argsParser.add_argument('--pheromone_evaporation_coefficient', default=0.4, type=float,
                            help="Define pheromone evaporation coefficient for aco")
    argsParser.add_argument('--pheromone_constant', default=1.0, type=float,
                            help="Define the pheromone constant for aco")
    argsParser.add_argument('--aco_iterations', default=300, type=int,
                            help="Define the number of iterations for aco")
    argsParser.add_argument('--aco_increasing_factor', default=0.9, type=float,
                            help="Define the increasing factor for aco. This will be applied in the process of the "
                                 "aco boost.")

    args = argsParser.parse_args()
    args = vars(args)

    #output
    print(' '.join(f'{k}={v}\n' for k, v in sorted(args.items())))

    return args
