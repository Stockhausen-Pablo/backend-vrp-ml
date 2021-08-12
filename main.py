import base64
import csv
import json
from datetime import datetime
from io import BytesIO
import src.Tour.TourManager as tManager
import redis
import flask

from timeit import default_timer as timer
from flask import Flask, send_file, make_response, render_template, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

from argsConfig import getParams
from src.Aco.AntManager import AntManager
from src.Mdp.VRPEnvironment import VRPEnvironment
from src.RL.Policy.PolicyManager import PolicyManager
from src.RL.VRPAgent import VRPAgent
from src.Tour.Stop import Stop
from src.Utils.helper import calculate_tour_meta
from src.Utils.memoryLoader import create_model_name
from src.Utils.plotter import plot_episode_stats, \
    plot_baseline_estimate, \
    plot_coordinates_with_coordinates_as_label, \
    plot_coordinates_with_stopnr__as_label, \
    plot_tour_with_stopnr_as_label, \
    plot_tours_individual

# GLOBAL VALUES
global parameter_groups

redis_host = "localhost"
redis_port = 6379
redis_password = ""


def event_stream_vrp_agent(red):
    pubsub = red.pubsub()
    pubsub.subscribe('updates')
    # TODO: handle client disconnection.
    for message in pubsub.listen():
        print(message)
        if message['type']=='message':
            yield 'data: %s\n\n' % message['data']


def load_stop_data(data_input):
    with open('data/stops/' + data_input + '.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            tManager.add_stop(
                Stop(str(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), int(row[6]),
                     int(row[7])))
    tManager.calculate_distance_matrix()
    tManager.init_capacity_demands()


def load_global_parameters():
    global parameter_groups

    with open("parameter_groups.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    parameter_groups = jsonObject


def main(args):
    # --------------------
    # ARGSPARSE
    # define parameters (the wall of parameters)

    # TRAINING
    num_episodes = args['num_episodes']
    max_steps = args['max_steps']

    # ML-PARAMETER
    learning_rate = args['learning_rate']
    stay_duration = args['stay_duration']
    discount_factor = args['discount_factor']
    exploration_factor = args['exploration_factor']
    ml_agent = args['agent']
    increasing_factor = args['increasing_factor']
    increasing_factor_good_episode = args['increasing_factor_good_episode']
    decreasing_factor = args['decreasing_factor']
    decreasing_factor_good_episode = args['decreasing_factor_good_episode']
    baseline_theta = args['baseline_theta']

    # THRESHOLD PARAMETERS
    # see argsConfig for help
    distance_utilization_threshold = args['distance_utilization_threshold']
    capacity_utilization_threshold = args['capacity_utilization_threshold']
    local_search_threshold = args['local_search_threshold']
    policy_reset_threshold = args['policy_reset_threshold']

    # ACO-PARAMETER
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
    data_input = input("Please specify the data source of the stops to be processed:") or 'short_train_data'
    print('-Regarding the Microhub name, this should be unique and used only for this Microhub.-')
    print('-The model of the agent is saved but also loaded based on the microhub names.-')
    microhub_name = input("Please specify the microhub name:") or "TestHub"
    shipper_name = input("Please specify the shipper name:") or "TestVersender"
    carrier_name = input("Please specify the carrier name:") or "TestCarrier"
    print('-Enter the delivery date. Possible Answers [Mon, Tue, Wed, Thurs, Fri, Sat]')
    delivery_date = input("Please specify the delivery date:") or "Test"
    amount_vehicles = int(input("How many vehicles will be used:") or 2)
    vehicle_speed = int(input("How fast is the vehicle [km/h]: ") or 30)
    capacity_weight = float(input("What is the maximum weight that the vehicle can carry:") or 180)
    capacity_volume = float(input("What is the maximum volume that the vehicle can hold:") or 500)

    # --------------------
    # SETTING UP TOUR MANAGER
    # Load Stop Data
    tManager.clear()
    load_stop_data(data_input)

    # Setup Distance Matrix for later use
    distance_matrix = tManager.get_distances()

    # --------------------
    # PLOT COORDINATES
    # overview of problem space (input)
    plot_coordinates_with_coordinates_as_label()
    plot_coordinates_with_stopnr__as_label()

    if args['train']:
        # --------------------TRAINING MODE--------------------
        print("-Entered Training Mode-")

        # --------------------
        # ANT COLONY OPTIMIZATION
        # setting up and running ACO
        print("-Starting up Ant Colony Optimization to get Probability Matrix-")
        antManager = AntManager(
            stops=tManager.get_list_of_stops(),
            start_stop=tManager.get_stop(0),
            vehicle_weight=capacity_weight,
            vehicle_volume=capacity_volume,
            vehicleCount=amount_vehicles,
            discount_alpha=aco_alpha_factor,
            discount_beta=aco_beta_factor,
            pheromone_evaporation_coefficient=pheromone_evaporation_coefficient,
            pheromone_constant=pheromone_constant,
            iterations=aco_iterations
        )

        # --------------------
        # RUN ACO
        # retrieving solution from ACO and preparing further transformation
        aco_start = timer()
        resultACO = antManager.run_aco()
        aco_end = timer()

        # --------------------
        # ACO RESULTS
        ant_shortest_distance = resultACO[0]
        ant_shortest_path = resultACO[1]
        aco_probability_Matrix = resultACO[2]

        # --------------------
        # ENVIRONMENT
        # setting up MDP-Environment
        print('SETTING UP ENVIRONMENT')
        environment = VRPEnvironment(
            states=tManager.get_list_of_stops(),
            # actions:
            # 0 = select microhub if tour full and possible Stops != null
            # 1 = select unvisited Node from possible Stops
            # 2 = select microhub if tour full and possible Stops = null
            actions=[0, 1, 2],
            distance_matrix=distance_matrix,
            microhub=tManager.get_microhub(),
            capacity_demands=tManager.get_capacity_demands_as_dict(),
            vehicles=amount_vehicles,
            vehicle_weight=capacity_weight,
            vehicle_volume=capacity_volume
        )

        # --------------------
        # POLICY NETWORK
        policyManager = PolicyManager(environment.get_all_state_hashes(),
                                      learning_rate,
                                      discount_factor,
                                      exploration_factor,
                                      increasing_factor,
                                      increasing_factor_good_episode,
                                      decreasing_factor,
                                      decreasing_factor_good_episode,
                                      baseline_theta,
                                      distance_utilization_threshold,
                                      capacity_utilization_threshold,
                                      local_search_threshold,
                                      policy_reset_threshold
                                      )

        # --------------------
        # LOAD PREVIOUS ML-MODEL
        print('LOADING MODEL')
        model_name = create_model_name(microhub_name, capacity_weight, capacity_volume, shipper_name, carrier_name,
                                       delivery_date, ml_agent)
        policyManager.loadModel(model_name)

        # --------------------
        # APPLY ACO TO ML-MODEL
        print('APPLYING ACO ON MODEL')
        policyManager.apply_aco_on_policy(aco_increasing_factor, aco_probability_Matrix)

        # --------------------
        # AGENT
        print('SETTING UP AGENT')
        agent = VRPAgent(env=environment,
                         policy_manager=policyManager,
                         num_episodes=num_episodes,
                         max_steps=max_steps,
                         discount_factor=discount_factor
                         )

        # --------------------
        # TRAINING RESULTS
        print('STARTING TRAINING')
        training_start = timer()
        episodeStatistics, policy_action_space, best_policy_reward, worst_policy_reward, last_policy_reward = agent.train_model()
        training_end = timer()
        current_policy_reward, final_tours = policyManager.construct_policy(policyManager.get_current_policy(),
                                                                            environment, max_steps)

        print("----------------------------------------")
        print("Best_policy_reward: ", best_policy_reward)
        print("Worst_policy_reward: ", worst_policy_reward)
        print("Final_policy_reward: ", last_policy_reward)
        print("ACO RUN TIME in s: ", (aco_end - aco_start))
        print("TRAINING RUN TIME in s: ", (training_end - training_start))
        # --------------------
        # PLOTTING TRAINING RESULTS
        plot_episode_stats(episodeStatistics, smoothing_window=25)
        plot_tour_with_stopnr_as_label(final_tours)
        current_baseline = policyManager.get_current_baseline_as_dict()
        plot_baseline_estimate(current_baseline)

        # --------------------
        # SAVING TRAINING RESULTS
        """
         Abbreviation explanation:
         w = weight
         v = volume
         s = shipper
         c = carrier
         d = delivery date
         a = agent
        """
        policyManager.saveModel(model_name)

    if args['test']:
        # --------------------TESTING MODE--------------------
        print("-Entered Testing Mode-")
        testing_start = timer()
        # --------------------
        # ENVIRONMENT
        # setting up MDP-Environment
        environment = VRPEnvironment(
            states=tManager.get_list_of_stops(),
            # actions:
            # 0 = select microhub if tour full and possible Stops != null
            # 1 = select unvisited Node from possible Stops
            # 2 = select microhub if tour full and possible Stops = null
            actions=[0, 1, 2],
            distance_matrix=distance_matrix,
            microhub=tManager.get_microhub(),
            capacity_demands=tManager.get_capacity_demands_as_dict(),
            vehicles=amount_vehicles,
            vehicle_weight=capacity_weight,
            vehicle_volume=capacity_volume
        )

        # --------------------
        # POLICY NETWORK
        policyManager = PolicyManager(environment.get_all_state_hashes(),
                                      learning_rate,
                                      discount_factor,
                                      exploration_factor,
                                      increasing_factor,
                                      increasing_factor_good_episode,
                                      decreasing_factor,
                                      decreasing_factor_good_episode,
                                      baseline_theta,
                                      distance_utilization_threshold,
                                      capacity_utilization_threshold,
                                      local_search_threshold,
                                      policy_reset_threshold
                                      )

        # --------------------
        # LOAD PREVIOUS ML-MODEL
        model_name = create_model_name(microhub_name, capacity_weight, capacity_volume, shipper_name, carrier_name,
                                       delivery_date, ml_agent)
        policyManager.loadModel(model_name)

        # --------------------
        # CONSTRUCTION SOLUTION
        current_policy_reward, final_tours = policyManager.construct_policy(policyManager.get_current_policy(),
                                                                            environment, max_steps)
        testing_end = timer()

        for tour in final_tours:
            for stop in tour:
                print(stop.tourStopId)

        # --------------------
        # CALCULATE/PRINT META TOUR CONSTRUCTION DATA
        total_box_amount = 0
        total_weight = 0.0
        total_volume = 0.0

        for tour in final_tours:
            for stop in tour:
                total_box_amount += stop.box_amount
                total_weight += stop.demand_weight
                total_volume += stop.demand_volume

        mean_box_amount = total_box_amount / len(final_tours)
        mean_volume = total_volume / len(final_tours)
        mean_weight = total_weight / len(final_tours)

        # total_time, total_distance, average_time_per_tour = calculate_delivery_time(vehicle_speed, stay_duration, final_tours)
        total_time, total_distance, average_time_per_tour, average_distance_per_tour = calculate_tour_meta(
            vehicle_speed, stay_duration, final_tours)

        print("Stop Amount: ", len(tManager.get_list_of_stops()))
        print("TESTING RUN TIME in s: ", (testing_end - testing_start))
        print("Amount of constructed Tours: ", len(final_tours))
        print("Mean Box Amount per Tour: ", mean_box_amount)
        print("Mean Volume per Tour: ", mean_volume)
        print("Mean Weight per Tour: ", mean_weight)
        print("Lost Volume per Tour: ", (capacity_volume - mean_volume))
        print("Lost Weight per Tour: ", (capacity_weight - mean_weight))
        print("Overall distance: ", total_distance)
        print("Mean Distance per Tour: ", (total_distance / len(final_tours)))
        print("Overall Time needed: ", total_time)
        print("Average Time needed per Tour: ", average_time_per_tour)

        # --------------------
        # PLOTTING TOUR (UNNECESSARY IN PRODUCTION)
        plot_tour_with_stopnr_as_label(final_tours)
        plot_tours_individual(final_tours, model_name)


def start_server(args):
    global parameter_groups

    # --------------------
    # SETTING UP TOUR MANAGER
    # Load Stop Data
    tManager.clear()
    load_stop_data(parameter_groups['Main-Settings']['data_input'])
    # Setup Distance Matrix for later use
    distance_matrix = tManager.get_distances()

    app = Flask(__name__)
    CORS(app)
    app.secret_key = 'asdf'
    red = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    api = Api(app)

    @app.route('/', methods=['GET'])
    def home():
        return "<h1>Distant Reading Archive</h1><p>Home</p>"

    @app.route('/parameters/groups', methods=['GET'])
    def get_parameter_groups():
        return flask.jsonify(parameter_groups)

    @app.route('/parameters', methods=['POST'])
    def parameters_update():
        parser = reqparse.RequestParser()
        parser.add_argument('data_input', required=True)
        parser.add_argument('microhub_name', required=True)
        parser.add_argument('shipper_name', required=True)
        parser.add_argument('carrier_name', required=True)
        parser.add_argument('delivery_date', required=True)
        parser.add_argument('amount_vehicles', required=True)
        parser.add_argument('vehicle_speed', required=True)
        parser.add_argument('capacity_weight', required=True)
        parser.add_argument('capacity_volume', required=True)

        args = parser.parse_args()

        global data_input
        global microhub_name
        global shipper_name
        global carrier_name
        global delivery_date
        global amount_vehicles
        global vehicle_speed
        global capacity_weight
        global capacity_volume

        data_input = args['data_input']
        microhub_name = args['microhub_name']
        shipper_name = args['shipper_name']
        carrier_name = args['carrier_name']
        delivery_date = args['delivery_date']
        amount_vehicles = args['amount_vehicles']
        vehicle_speed = args['vehicle_speed']
        capacity_weight = args['capacity_weight']
        capacity_volume = args['capacity_volume']

        return {'POST': "succesful"}, 200

    @app.route('/images/render/plt-coord-with-coord', methods=['GET'])
    def images_render_plt_coord_with_coord():
        plt = plot_coordinates_with_coordinates_as_label()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return '<img src="data:image/png;base64,{}">'.format(plot_url)

    @app.route('/images/render/plt-coord-with-stop-nr', methods=['GET'])
    def images_render_plt_coord_with_stop_nr():
        plt = plot_coordinates_with_stopnr__as_label()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return '<img src="data:image/png;base64,{}">'.format(plot_url)

    @app.route('/ml-service/training/stream', methods=['GET'])
    def ml_service_training_stream():
        return flask.Response(event_stream_vrp_agent(red),
                              mimetype="text/event-stream")

    @app.route('/ml-service/training/update', methods=['POST'])
    def ml_service_training_update():
        epoch = flask.request.form['epoch']
        policy_reward = flask.request.form['policy_reward']
        sum_G_t = flask.request.form['sum_G_t']
        best_policy_reward = flask.request.form['best_policy_reward']
        worst_policy_reward = flask.request.form['worst_policy_reward']

        now = datetime.now().replace(microsecond=0).time()
        red.publish('updates', u'[%s] %s, %s, %s, %s, %s' % (now, epoch, policy_reward, sum_G_t, best_policy_reward, worst_policy_reward))
        return flask.Response(status=204)

    @app.route('/ml-service/training/start', methods=['GET'])
    def ml_service_training_start():
        # --------------------TRAINING MODE--------------------
        print("-Entered Training Mode-")

        # --------------------
        # ANT COLONY OPTIMIZATION
        # setting up and running ACO
        print("-Starting up Ant Colony Optimization to get Probability Matrix-")
        antManager = AntManager(
            stops=tManager.get_list_of_stops(),
            start_stop=tManager.get_stop(0),
            vehicle_weight=parameter_groups['Main-Settings']['capacity_weight'],
            vehicle_volume=parameter_groups['Main-Settings']['capacity_volume'],
            vehicleCount=parameter_groups['Main-Settings']['amount_vehicles'],
            discount_alpha=parameter_groups['ACO-Settings']['aco_alpha_factor'],
            discount_beta=parameter_groups['ACO-Settings']['aco_beta_factor'],
            pheromone_evaporation_coefficient=parameter_groups['ACO-Settings']['pheromone_evaporation_coefficient'],
            pheromone_constant=parameter_groups['ACO-Settings']['pheromone_constant'],
            iterations=parameter_groups['ACO-Settings']['aco_iterations']
        )

        # --------------------
        # RUN ACO
        # retrieving solution from ACO and preparing further transformation
        aco_start = timer()
        resultACO = antManager.run_aco()
        aco_end = timer()

        # --------------------
        # ACO RESULTS
        ant_shortest_distance = resultACO[0]
        ant_shortest_path = resultACO[1]
        aco_probability_Matrix = resultACO[2]

        # --------------------
        # ENVIRONMENT
        # setting up MDP-Environment
        print('SETTING UP ENVIRONMENT')
        environment = VRPEnvironment(
            states=tManager.get_list_of_stops(),
            # actions:
            # 0 = select microhub if tour full and possible Stops != null
            # 1 = select unvisited Node from possible Stops
            # 2 = select microhub if tour full and possible Stops = null
            actions=[0, 1, 2],
            distance_matrix=distance_matrix,
            microhub=tManager.get_microhub(),
            capacity_demands=tManager.get_capacity_demands_as_dict(),
            vehicles=parameter_groups['Main-Settings']['amount_vehicles'],
            vehicle_weight=parameter_groups['Main-Settings']['capacity_weight'],
            vehicle_volume=parameter_groups['Main-Settings']['capacity_volume']
        )

        # --------------------
        # POLICY NETWORK
        policyManager = PolicyManager(environment.get_all_state_hashes(),
                                      parameter_groups['ML-Settings']['learning_rate'],
                                      parameter_groups['ML-Settings']['discount_factor'],
                                      parameter_groups['ML-Settings']['exploration_factor'],
                                      parameter_groups['Additional-ML-Settings']['increasing_factor'],
                                      parameter_groups['Additional-ML-Settings']['increasing_factor_good_episode'],
                                      parameter_groups['Additional-ML-Settings']['decreasing_factor'],
                                      parameter_groups['Additional-ML-Settings']['decreasing_factor_good_episode'],
                                      parameter_groups['ML-Settings']['baseline_theta'],
                                      parameter_groups['Threshold-Settings']['distance_utilization_threshold'],
                                      parameter_groups['Threshold-Settings']['capacity_utilization_threshold'],
                                      parameter_groups['Threshold-Settings']['local_search_threshold'],
                                      parameter_groups['Threshold-Settings']['policy_reset_threshold']
                                      )

        # --------------------
        # LOAD PREVIOUS ML-MODEL
        print('LOADING MODEL')
        model_name = create_model_name(parameter_groups['Main-Settings']['microhub_name'],
                                       parameter_groups['Main-Settings']['capacity_weight'],
                                       parameter_groups['Main-Settings']['capacity_volume'],
                                       parameter_groups['Main-Settings']['shipper_name'],
                                       parameter_groups['Main-Settings']['carrier_name'],
                                       parameter_groups['Main-Settings']['delivery_date'],
                                       parameter_groups['ML-Settings']['ml_agent'])
        policyManager.loadModel(model_name)

        # --------------------
        # APPLY ACO TO ML-MODEL
        print('APPLYING ACO ON MODEL')
        policyManager.apply_aco_on_policy(parameter_groups['ACO-Settings']['aco_increasing_factor'],
                                          aco_probability_Matrix)

        # --------------------
        # AGENT
        print('SETTING UP AGENT')
        agent = VRPAgent(env=environment,
                         policy_manager=policyManager,
                         num_episodes=parameter_groups['Training-Settings']['num_episodes'],
                         max_steps=parameter_groups['Training-Settings']['max_steps'],
                         discount_factor=parameter_groups['ML-Settings']['discount_factor']
                         )

        # --------------------
        # TRAINING RESULTS
        print('STARTING TRAINING')
        training_start = timer()
        episodeStatistics, policy_action_space, best_policy_reward, worst_policy_reward, last_policy_reward = agent.train_model()
        training_end = timer()
        current_policy_reward, final_tours = policyManager.construct_policy(policyManager.get_current_policy(),
                                                                            environment, parameter_groups['Training-Settings']['max_steps'])

        print("----------------------------------------")
        print("Best_policy_reward: ", best_policy_reward)
        print("Worst_policy_reward: ", worst_policy_reward)
        print("Final_policy_reward: ", last_policy_reward)
        print("ACO RUN TIME in s: ", (aco_end - aco_start))
        print("TRAINING RUN TIME in s: ", (training_end - training_start))
        # --------------------
        # PLOTTING TRAINING RESULTS
        plot_episode_stats(episodeStatistics, smoothing_window=25)
        plot_tour_with_stopnr_as_label(final_tours)
        current_baseline = policyManager.get_current_baseline_as_dict()
        plot_baseline_estimate(current_baseline)

        # --------------------
        # SAVING TRAINING RESULTS
        """
         Abbreviation explanation:
         w = weight
         v = volume
         s = shipper
         c = carrier
         d = delivery date
         a = agent
        """
        policyManager.saveModel(model_name)

    @app.route('/ml-service/testing', methods=['GET'])
    def ml_service_testing():
        return "<h1>Distant Reading Archive</h1><p>Home</p>"

    app.debug = True
    app.run(threaded=True)


if __name__ == "__main__":
    args = getParams()
    load_global_parameters()
    #main(args)
    start_server(args)
