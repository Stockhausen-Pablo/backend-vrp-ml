# Import the argparse library
import argparse

def getParams():
    # Creating the parser
    argsParser = argparse.ArgumentParser(
        description='Reinforcment Learning by using a Markov Decision process to solve VRP')

    # Add arguments
    # ------------------
    # Training
    argsParser.add_argument('--train', default=True, action='store_true', help='entering training mode')

    # Testing
    argsParser.add_argument('--test', default=False, action='store_false', help="entering test mode")

    # Agent
    argsParser.add_argument('--agent', default='policy_gradient_agent', help="Use the Policy Gradient Agent")
    argsParser.add_argument('--actor_learning_rate', default=1e-4, type=float,
                            help="Define actor learning rate")
    argsParser.add_argument('--critic_learning_rate', default=1e-4, type=float,
                            help="Define critic learning rate")

    args = argsParser.parse_args()
    args = vars(args)

    #output
    print(' '.join(f'{k}={v}' for k, v in sorted(args.items())))

    return args
