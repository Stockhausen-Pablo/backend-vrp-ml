import pandas as pd

import src.Tour.TourManager as tManager
from src.Aco.Ant import Ant


class AntManager:
    def __init__(self,
                 stops,
                 start_stop,
                 vehicleCount,
                 vehicle_weight,
                 vehicle_volume,
                 discount_alpha,
                 discount_beta,
                 pheromone_evaporation_coefficient,
                 pheromone_constant,
                 iterations):

        # --------------------
        # NODES
        self.nodes = stops
        self.state_hashes = [node.hash_id for node in self.nodes]
        self.nodes_amount = len(stops)
        self.start_stop = 0 if start_stop is None else start_stop
        self.microhub_hash = self.start_stop.hash_id
        self.microhub_counter = 0

        self.first_run = True

        # --------------------
        # PHEROMONE
        self.tour_size = len(stops)
        self.df_pheromone_matrix = self.setup_pheromone_matrix()
        self.df_updated_pheromone_matrix = self.setup_pheromone_matrix()

        # --------------------
        # ACO-PARAMETER
        self.discountAlpha = discount_alpha
        self.discountBeta = discount_beta
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations

        # --------------------
        # LIST OF ANTS
        self.antCount = vehicleCount
        self.antWeight = vehicle_weight
        self.antVolume = vehicle_volume
        self.ants = self.set_ants(self.start_stop)

        # --------------------
        # ACO-SOLUTION
        # self.ant_probability_Matrix = load_memory_df_from_local('./probabilityMatrixByAnts.pkl', self.nodes)
        self.ant_probability_Matrix = self.setup_ant_probability_matrix()
        self.shortest_distance = None
        self.shortest_path = None
        self.winner_ant = None

    def set_ants(self, startStop: object) -> object:
        """
        Constructs the ants with the number of vehicles.
        :param startStop: point of departure
        :return: Ant objects
        """
        print("-Setting up Ants-")
        if self.first_run:
            return [Ant(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromone_matrix,
                        self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, first_run=True)
                    for _ in range(self.antCount)]

        for ant in self.ants:
            ant.__init__(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromone_matrix,
                         self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, first_run=False)

    def setup_ant_probability_matrix(self) -> object:
        """
        :return: ant probability matrix
        """
        df_new_ant_probability_matrix = pd.DataFrame(index=self.state_hashes[1:], columns=self.state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, 0))
        df_new_pickle = df_new_ant_probability_matrix.append(new_row, ignore_index=False)
        df_new_pickle['{}/{}'.format(self.microhub_hash, 0)] = 0.0
        df_new_ant_probability_matrix.fillna(value=0.0, inplace=True)
        return df_new_ant_probability_matrix

    def setup_pheromone_matrix(self) -> object:
        """
        :return: pheromone matrix
        """
        df_new_pheromone_matrix = pd.DataFrame(index=self.state_hashes[1:], columns=self.state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, 0))
        df_new_pheromone_matrix = df_new_pheromone_matrix.append(new_row, ignore_index=False)
        df_new_pheromone_matrix['{}/{}'.format(self.microhub_hash, 0)] = 0.0
        df_new_pheromone_matrix.fillna(value=0.0, inplace=True)
        return df_new_pheromone_matrix

    def update_probability_matrix(self) -> object:
        """
        Updates the probability matrix of the ant manager.
        :return: None
        """
        print('-updating aco probability matrix-')
        sum_all_probabilities = 0.0
        for i in self.ant_probability_Matrix:
            for j in self.ant_probability_Matrix:
                # (\tau_{i,j})^\alpha
                tau_i_j = float(self.df_pheromone_matrix.at[i, j])
                # (\eta_{i,j})}^\beta
                stop_row = next((node for node in self.nodes if node.hash_id == i), None)
                stop_col = next((node for node in self.nodes if node.hash_id == j), None)
                eta_i_j = float(tManager.get_distance_by_matrix(stop_row.hash_id, stop_col.hash_id))
                # get the sum of all probabilities
                sum_all_probabilities += (tau_i_j * eta_i_j)
            break

        for i in self.df_pheromone_matrix:
            self.microhub_counter = 0
            for j in self.df_pheromone_matrix:
                if i not in self.ant_probability_Matrix.index.values:
                    new_row = pd.Series(name=j)
                    self.ant_probability_Matrix = self.ant_probability_Matrix.append(new_row, ignore_index=False)
                    self.ant_probability_Matrix[j] = 0.0
                    self.ant_probability_Matrix.fillna(value=0.0, inplace=True)
                if j not in self.ant_probability_Matrix.index.values:
                    new_row = pd.Series(name=j)
                    self.ant_probability_Matrix = self.ant_probability_Matrix.append(new_row, ignore_index=False)
                    self.ant_probability_Matrix[j] = 0.0
                    self.ant_probability_Matrix.fillna(value=0.0, inplace=True)
                # (\tau_{i,j})^\alpha
                tau_i_j = self.df_pheromone_matrix.at[i, j]
                # (\eta_{i,j})}^\beta
                i_hash_comparer = i
                j_hash_comparer = j
                if "/" in str(i_hash_comparer):
                    i_hash_comparer = self.microhub_hash
                if "/" in str(j_hash_comparer):
                    j_hash_comparer = self.microhub_hash
                stop_row = next((node for node in self.nodes if node.hash_id == i_hash_comparer), None)
                stop_col = next((node for node in self.nodes if node.hash_id == j_hash_comparer), None)
                eta_i_j = tManager.get_distance_by_matrix(stop_row.hash_id, stop_col.hash_id)
                # calculate Probability
                probability_i_j = float(((tau_i_j * eta_i_j) / sum_all_probabilities))
                # set Probability
                self.ant_probability_Matrix.at[i, j] = float(probability_i_j)

    def update_pheromone_matrix(self) -> object:
        """
        Updates the pheromone matrix of the ant manager.
        :return: None
        """
        print('-updating pheromone matrix-')
        for df_start in self.df_updated_pheromone_matrix:
            self.microhub_counter = 0
            for df_end in self.df_updated_pheromone_matrix:
                if df_end not in self.df_pheromone_matrix.index.values:
                    new_row = pd.Series(name=df_end)
                    self.df_pheromone_matrix = self.df_pheromone_matrix.append(new_row, ignore_index=False)
                    self.df_pheromone_matrix[df_end] = 0.0
                    self.df_pheromone_matrix.fillna(value=0.0, inplace=True)

                self.df_pheromone_matrix.at[df_start, df_end] *= (1 - self.pheromone_evaporation_coefficient)
                self.df_pheromone_matrix.at[df_start, df_end] += self.df_updated_pheromone_matrix.at[df_start, df_end]
        self.df_pheromone_matrix.reset_index()
        print('done updating pheromone matrix')

    def update_pheromone_matrix_by_ant_solution(self, ant: object) -> object:
        """
        Updates the pheromone matrix of the ant manager by the respective ant solution.
        :param ant: Ant object
        :return: None
        """
        print('-updating pheromone matrix based on ant solution-')
        tours = ant.get_all_tours()
        self.microhub_counter = 0
        for tour in tours:
            for index, stop in enumerate(tour):
                microhub_exists = False
                stop_b = 0
                if index == len(tour) - 1:
                    break
                    # stop_b = tour[0]
                else:
                    stop_b = tour[index + 1]

                stop_hash = stop.hash_id
                stop_b_hash = stop_b.hash_id

                if stop_hash == self.microhub_hash:
                    stop_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
                if stop_b_hash == self.microhub_hash:
                    self.microhub_counter += 1
                    stop_b_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
                    microhub_exists = True

                new_pheromone_value = float(self.pheromone_constant / ant.get_travelled_distance())  # 1/Total Length

                if microhub_exists:
                    if ('{}/{}'.format(self.microhub_hash,
                                       self.microhub_counter) not in self.df_updated_pheromone_matrix.index.values):
                        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                        self.df_updated_pheromone_matrix = self.df_updated_pheromone_matrix.append(new_row,
                                                                                                   ignore_index=False)
                        self.df_updated_pheromone_matrix[
                            '{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
                        self.df_updated_pheromone_matrix.fillna(value=0.0, inplace=True)
                    self.df_updated_pheromone_matrix.at[stop_hash, stop_b_hash] += new_pheromone_value
                    self.df_updated_pheromone_matrix.at[stop_b_hash, stop_hash] += new_pheromone_value
                else:
                    self.df_updated_pheromone_matrix.at[stop_hash, stop_b_hash] += new_pheromone_value
                    self.df_updated_pheromone_matrix.at[stop_b_hash, stop_hash] += new_pheromone_value
            self.df_updated_pheromone_matrix.reset_index()
            # self.updated_pheromoneMatrix[stop.stopid][stop_b.stopid] += new_pheromone_value
            # self.updated_pheromoneMatrix[stop_b.stopid][stop.stopid] += new_pheromone_value

    def run_aco(self) -> object:
        """
        Runs the whole Ant colony optimization. Main Method of the class.
        :return: shortest overall distance, shortest path (tours), updated ant probability matrix
        """
        print("-Running Colony Optimization-")
        for iteration in range(self.iterations):
            for ant in self.ants:
                ant.move_ant()
            self.ants = sorted(self.ants, key=lambda ant: ant.get_travelled_distance())
            for ant in self.ants:
                self.update_pheromone_matrix_by_ant_solution(ant)
                if not self.shortest_distance:
                    self.shortest_distance = ant.get_travelled_distance()
                if not self.shortest_path:
                    self.shortest_path = ant.get_all_tours()
                if ant.get_travelled_distance() < self.shortest_distance:
                    self.shortest_distance = ant.get_travelled_distance()
                    self.shortest_path = ant.get_all_tours()
                    self.winner_ant = ant

            self.update_pheromone_matrix()

            if self.first_run:
                self.first_run = False

            self.set_ants(self.start_stop)
            # self.updated_pheromoneMatrix = [[0.0] * self.tour_size for _ in range(self.tour_size)]
            for col in self.df_updated_pheromone_matrix.columns:
                self.df_updated_pheromone_matrix[col].values[:] = 0.0
            print('{0}/{1} Searching...'.format(iteration + 1, self.iterations))

        self.update_probability_matrix()

        return self.shortest_distance, self.shortest_path, self.ant_probability_Matrix
