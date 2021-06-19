import csv

import src.Tour.TourManager as tManager
from argsConfig import getParams
from src.Aco.AntManager import AntManager
from src.Mdp.VRPEnvironment import VRPEnvironment
from src.RL.Policy.PolicyManager import PolicyManager
from src.RL.VRPAgent import VRPAgent
from src.Tour.Stop import Stop
from src.Utils.helper import normalize_df
from src.Utils.memoryLoader import create_model_name
from src.Utils.plotter import plot_episode_stats, plot_baseline_estimate, plotCoordinatesWithCoordinatesLabel, plotCoordinatesWithStopNrLabel, plotTourWithStopNrLabel


def loadStopData(dataSet):
    with open('data/stops/' + dataSet + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tManager.addStop(
                Stop(float(row[0]), int(row[1]), float(row[2]), float(row[3]), int(row[4]), int(row[5])))
    tManager.calculateDistanceMatrix()
    tManager.initCapacityDemands()


def main(args):
    # --------------------
    # ARGSPARSE
    # define parameters
    learning_rate = args['learning_rate']
    discount_factor = args['discount_factor']
    exploration_factor = args['exploration_factor']
    num_episodes = args['num_episodes']
    max_steps = args['max_steps']
    ml_agent = args['agent']
    # aco settings
    aco_alpha_factor = args['aco_alpha_factor']
    aco_beta_factor = args['aco_beta_factor']
    pheromone_evaporation_coefficient = args['pheromone_evaporation_coefficient']
    pheromone_constant = args['pheromone_constant']
    aco_iterations = args['aco_iterations']
    aco_increasing_factor = args['aco_increasing_factor']
    # --------------------
    # INPUT
    # define meta data
    print("---------System menu---------")
    print("Below please specify the configuration options of the program")
    dataSet = input("Please specify the data source of the stops to be processed:") or 'test'
    print('-Regarding the Microhub name, this should be unique and used only for this Microhub.-')
    print('-The model of the agent is saved but also loaded based on the microhub names.-')
    microhub_name = input("Please specify the microhub name:") or "Prenzlauer_Berg"
    shipper_name = input("Please specify the shipper name:") or "Brodowin"
    carrier_name = input("Please specify the carrier name:") or "Urban_Cargo"
    amount_vehicles = int(input("How many vehicles will be used:"))
    capacity_weight = float(input("What is the maximum weight that the vehicle can carry:"))
    capacity_volume = float(input("What is the maximum volume that the vehicle can hold:"))

    # --------------------
    # SETTING UP TOUR MANAGER
    # Load Stop Data
    tManager.clear()
    loadStopData(dataSet)

    # Setup Distance Matrix for later use
    distance_matrix = tManager.getDistances()

    # --------------------
    # PLOT COORDINATES
    # overview of problem space (input)
    plotCoordinatesWithCoordinatesLabel()
    plotCoordinatesWithStopNrLabel()

    if args['train']:
        # --------------------TRAINING MODE--------------------
        print("-Entered Training Mode-")

        # --------------------
        # ANT COLONY OPTIMIZATION
        # setting up and running ACO
        print("-Starting up Ant Colony Optimization to get Probability Matrix-")
        antManager = AntManager(
            stops=tManager.getListOfStops(),
            start_stop=tManager.getStop(0),
            vehicleWeight=capacity_weight,
            vehicleVolume=capacity_volume,
            vehicleCount=amount_vehicles,
            discountAlpha=aco_alpha_factor,
            discountBeta=aco_beta_factor,
            pheromone_evaporation_coefficient=pheromone_evaporation_coefficient,
            pheromone_constant=pheromone_constant,
            iterations=aco_iterations
        )

        # --------------------
        # ACO RESULTS
        # retrieving solution from ACO and preparing further transformation
        resultACO = antManager.runACO()
        ant_shortest_distance = resultACO[0]
        ant_shortest_path = resultACO[1]
        aco_probability_Matrix = resultACO[2]

        # --------------------
        # NORMALIZING PROBABILITIES
        # normalize the ant probability Matrix
        normalized_aco_probability_Matrix = normalize_df(aco_probability_Matrix)

        # --------------------
        # ENVIRONMENT
        # setting up MDP-Environment
        environment = VRPEnvironment(
            states=tManager.getListOfStops(),
            # actions:
            # 0 = select microhub if tour full and possible Stops != null
            # 1 = select unvisited Node from possible Stops
            # 2 = select microhub if tour full and possible Stops = null
            actions=[0, 1, 2],
            probabilityMatrix=normalized_aco_probability_Matrix,
            distanceMatrix=distance_matrix,
            microHub=tManager.getMicrohub(),
            capacityDemands=tManager.getCapacityDemands(),
            vehicles=amount_vehicles,
            vehicleWeight=capacity_weight,
            vehicleVolume=capacity_volume
        )

        # --------------------
        # POLICY NETWORK
        policyManager = PolicyManager(environment.getStateHashes(),
                                      learning_rate,
                                      discount_factor,
                                      exploration_factor
                                      )

        # --------------------
        # LOAD PREVIOUS ML-MODEL
        model_name = create_model_name(microhub_name, capacity_weight, capacity_volume, shipper_name, carrier_name, ml_agent)
        policyManager.loadModel(model_name)

        # --------------------
        # APPLY ACO TO ML-MODEL
        policyManager.apply_aco_on_policy(aco_increasing_factor, normalized_aco_probability_Matrix)

        # --------------------
        # AGENT
        agent = VRPAgent(env=environment,
                         policyManager=policyManager,
                         num_episodes=num_episodes,
                         max_steps=max_steps,
                         discount_factor=discount_factor
                         )

        # --------------------
        # TRAINING RESULTS
        episodeStatistics, policy_action_space, best_policy_reward, last_policy_reward = agent.train_model()
        current_policy_reward, final_tours = policyManager.construct_policy(policyManager.get_current_policy(), environment, max_steps)

        print("----------------------------------------")
        print("Best_policy_reward: ", best_policy_reward)
        print("Final_policy_reward: ", last_policy_reward)

        # --------------------
        # PLOTTING TRAINING RESULTS
        plot_episode_stats(episodeStatistics, smoothing_window=25)
        plotTourWithStopNrLabel(final_tours)
        current_baseline = policyManager.get_current_baseline_as_dict()
        plot_baseline_estimate(current_baseline)

        # --------------------
        # SAVING TRAINING RESULTS
        # w = weight
        # v = volume
        # s = shipper
        # c = carrier
        # a = agent
        policyManager.saveModel(model_name)


        if args['test']:
            # --------------------TESTING MODE--------------------
            print("-Entered Testing Mode-")


if __name__ == "__main__":
    args = getParams()
    main(args)
