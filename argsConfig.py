# Import the argparse library
import argparse


def getParams():
    # Creating the parser
    argsParser = argparse.ArgumentParser(
        description='Reinforcment Learning by using a MDP/REINFORCE to solve CVRP')

    # Add arguments
    # ------------------
    # Convert
    argsParser.add_argument('--convert', default=False, action='store_true', help='entering convert mode')

    # Training
    argsParser.add_argument('--train', default=True, action='store_true', help='entering training mode')
    argsParser.add_argument('--num_episodes', default=100, type=int,
                            help="Define number of episodes for the training process")
    argsParser.add_argument('--max_steps', default=10000, type=int,
                            help="Define the number of maximal steps that can be taking for the training process")

    # Testing
    argsParser.add_argument('--test', default=False, action='store_false', help="entering test mode")

    # Agent
    argsParser.add_argument('--agent', default='policy_gradient_agent', help="Use the Policy Gradient Agent")

    argsParser.add_argument('--learning_rate', default=2 ** -3, type=float,
                            help="Define learning rate")

    argsParser.add_argument('--discount_factor', default=0.95, type=float,
                            help="Define discount factor")

    argsParser.add_argument('--exploration_factor', default=0.05, type=float,
                            help="Define exploration factor")

    # ACO
    argsParser.add_argument('--aco_alpha_factor', default=0.5, type=float,
                            help="Define discount alpha for aco")
    argsParser.add_argument('--aco_beta_factor', default=1.2, type=float,
                            help="Define discount beta for aco")
    argsParser.add_argument('--pheromone_evaporation_coefficient', default=0.4, type=float,
                            help="Define pheromone evaporation coefficient for aco")
    argsParser.add_argument('--pheromone_constant', default=1.0, type=float,
                            help="Define the pheromone constant for aco")
    argsParser.add_argument('--aco_iterations', default=80, type=int,
                            help="Define the number of iterations for aco")

    args = argsParser.parse_args()
    args = vars(args)

    #output
    print(' '.join(f'{k}={v}\n' for k, v in sorted(args.items())))

    return args
