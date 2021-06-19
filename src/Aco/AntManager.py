import pandas as pd

import src.Tour.TourManager as tManager

from src.Aco.Ant import Ant
from src.Utils.memoryLoader import load_memory_df_from_local, save_memory_df_to_local


class AntManager:

    def __init__(self,
                 stops,
                 start_stop,
                 vehicleCount,
                 vehicleWeight,
                 vehicleVolume,
                 discountAlpha,
                 discountBeta,
                 pheromone_evaporation_coefficient,
                 pheromone_constant,
                 iterations):

        # Initializing nodes
        self.nodes = stops
        self.state_hashes = [node.hashIdentifier for node in self.nodes]
        self.nodes_amount = len(stops)
        self.start_stop = 0 if start_stop is None else start_stop
        self.microhub_hash = self.start_stop.hashIdentifier
        self.microhub_counter = 0

        self.firstInit = True

        # Initializing Pheromone
        self.tour_size = len(stops)
        self.df_pheromoneMatrix = self.setup_pheromoneMatrix()
        self.df_updated_pheromoneMatrix = self.setup_pheromoneMatrix()
        # self.load_df_pheromoneMatrix()
        # self.setup_df_updated_pheromoneMatrix()
        # self.pheromoneMatrix = [
        #    [0.0] * self.tour_size for _ in range(self.tour_size)]
        # self.updated_pheromoneMatrix = [
        #    [0.0] * self.tour_size for _ in range(self.tour_size)]

        self.discountAlpha = discountAlpha
        self.discountBeta = discountBeta
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations

        # Initializing the List of Ants
        self.antCount = vehicleCount
        self.antWeight = vehicleWeight
        self.antVolume = vehicleVolume
        self.ants = self.setAnts(self.start_stop)

        # For Solution
        #self.ant_probability_Matrix = load_memory_df_from_local('./probabilityMatrixByAnts.pkl', self.nodes)
        self.ant_probability_Matrix = self.setup_antProbabilyMatrix()
        self.shortest_distance = None
        self.shortest_path = None
        self.winner_ant = None

    def setAnts(self, startStop):
        print("-Setting up Ants-")
        if self.firstInit:
            return [Ant(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromoneMatrix,
                        self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, firstInit=True)
                    for _ in range(self.antCount)]

        for ant in self.ants:
            ant.__init__(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromoneMatrix,
                         self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, firstInit=False)

    def setup_antProbabilyMatrix(self):
        df_new_antProbabilyMatrix = pd.DataFrame(index=self.state_hashes[1:], columns=self.state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, 0))
        df_new_pickle = df_new_antProbabilyMatrix.append(new_row, ignore_index=False)
        df_new_pickle['{}/{}'.format(self.microhub_hash, 0)] = 0.0
        df_new_antProbabilyMatrix.fillna(value=0.0, inplace=True)
        return df_new_antProbabilyMatrix

    def setup_pheromoneMatrix(self):
        df_new_pheromoneMatrix = pd.DataFrame(index=self.state_hashes[1:], columns=self.state_hashes[1:])
        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, 0))
        df_new_pheromoneMatrix = df_new_pheromoneMatrix.append(new_row, ignore_index=False)
        df_new_pheromoneMatrix['{}/{}'.format(self.microhub_hash, 0)] = 0.0
        df_new_pheromoneMatrix.fillna(value=0.0, inplace=True)
        return df_new_pheromoneMatrix

    def updateProbabilityMatrix(self):
        sum_all_probabilities = 0.0
        for i in self.ant_probability_Matrix:
            for j in self.ant_probability_Matrix:
                # (\tau_{i,j})^\alpha
                tau_i_j = float(self.df_pheromoneMatrix.at[i, j])
                # (\eta_{i,j})}^\beta
                stop_row = next((node for node in self.nodes if node.hashIdentifier == i), None)
                stop_col = next((node for node in self.nodes if node.hashIdentifier == j), None)
                eta_i_j = float(tManager.getDistanceByMatrix(stop_row.hashIdentifier, stop_col.hashIdentifier))
                # get the sum of all probabilities
                sum_all_probabilities += (tau_i_j * eta_i_j)
            break

        for i in self.df_pheromoneMatrix:
            self.microhub_counter = 0
            for j in self.df_pheromoneMatrix:
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
                tau_i_j = self.df_pheromoneMatrix.at[i, j]
                # (\eta_{i,j})}^\beta
                i_hashComparer = i
                j_hashComparer = j
                if "/" in str(i_hashComparer):
                    i_hashComparer = self.microhub_hash
                if "/" in str(j_hashComparer):
                    j_hashComparer = self.microhub_hash
                stop_row = next((node for node in self.nodes if node.hashIdentifier == i_hashComparer), None)
                stop_col = next((node for node in self.nodes if node.hashIdentifier == j_hashComparer), None)
                eta_i_j = tManager.getDistanceByMatrix(stop_row.hashIdentifier, stop_col.hashIdentifier)
                # calculate Probability
                probability_i_j = float(((tau_i_j * eta_i_j) / sum_all_probabilities))
                # set Probability
                self.ant_probability_Matrix.at[i, j] = float(probability_i_j)

    def updatePheromoneMatrix(self):
        for df_start in self.df_updated_pheromoneMatrix:
            self.microhub_counter = 0
            for df_end in self.df_updated_pheromoneMatrix:
                microhubExists = False

                if df_end not in self.df_pheromoneMatrix.index.values:
                    new_row = pd.Series(name=df_end)
                    self.df_pheromoneMatrix = self.df_pheromoneMatrix.append(new_row, ignore_index=False)
                    self.df_pheromoneMatrix[df_end] = 0.0
                    self.df_pheromoneMatrix.fillna(value=0.0, inplace=True)

                self.df_pheromoneMatrix.at[df_start, df_end] *= (1 - self.pheromone_evaporation_coefficient)
                self.df_pheromoneMatrix.at[df_start, df_end] += self.df_updated_pheromoneMatrix.at[df_start, df_end]

                if microhubExists:
                    self.microhub_counter += 1

    def invalid_updatePheromoneMatrix(self):
        for start in range(len(self.pheromoneMatrix)):
            for end in range(len(self.pheromoneMatrix)):
                self.pheromoneMatrix[start][end] *= (1 - self.pheromone_evaporation_coefficient)
                self.pheromoneMatrix[start][end] += self.updated_pheromoneMatrix[start][end]

    def updatePheromoneMatrixByAnt(self, ant):
        tours = ant.getTours()
        self.microhub_counter = 0
        for tour in tours:
            for index, stop in enumerate(tour):
                microhubExists = False
                stop_b = 0
                if index == len(tour) - 1:
                    break
                    # stop_b = tour[0]
                else:
                    stop_b = tour[index + 1]

                stop_hash = stop.hashIdentifier
                stop_b_hash = stop_b.hashIdentifier

                if stop_hash == self.microhub_hash:
                    stop_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
                if stop_b_hash == self.microhub_hash:
                    self.microhub_counter += 1
                    stop_b_hash = '{}/{}'.format(self.microhub_hash, self.microhub_counter)
                    microhubExists = True

                new_pheromoneValue = float(self.pheromone_constant / ant.get_travelled_Distance())  # 1/Total Length

                if microhubExists:
                    if ('{}/{}'.format(self.microhub_hash,
                                       self.microhub_counter) not in self.df_updated_pheromoneMatrix.index.values):
                        new_row = pd.Series(name='{}/{}'.format(self.microhub_hash, self.microhub_counter))
                        self.df_updated_pheromoneMatrix = self.df_updated_pheromoneMatrix.append(new_row, ignore_index=False)
                        self.df_updated_pheromoneMatrix['{}/{}'.format(self.microhub_hash, self.microhub_counter)] = 0.0
                        self.df_updated_pheromoneMatrix.fillna(value=0.0, inplace=True)
                    self.df_updated_pheromoneMatrix.at[stop_hash, stop_b_hash] += new_pheromoneValue
                    self.df_updated_pheromoneMatrix.at[stop_b_hash, stop_hash] += new_pheromoneValue
                else:
                    self.df_updated_pheromoneMatrix.at[stop_hash, stop_b_hash] += new_pheromoneValue
                    self.df_updated_pheromoneMatrix.at[stop_b_hash, stop_hash] += new_pheromoneValue
                # self.updated_pheromoneMatrix[stop.stopid][stop_b.stopid] += new_pheromoneValue
                # self.updated_pheromoneMatrix[stop_b.stopid][stop.stopid] += new_pheromoneValue

    def runACO(self):
        print("-Running Colony Optimization-")
        for iteration in range(self.iterations):
            for ant in self.ants:
                ant.moveAnt()
            self.ants = sorted(self.ants, key=lambda ant: ant.get_travelled_Distance())
            for ant in self.ants:
                self.updatePheromoneMatrixByAnt(ant)
                if not self.shortest_distance:
                    self.shortest_distance = ant.get_travelled_Distance()
                if not self.shortest_path:
                    self.shortest_path = ant.getTours()
                if ant.get_travelled_Distance() < self.shortest_distance:
                    self.shortest_distance = ant.get_travelled_Distance()
                    self.shortest_path = ant.getTours()
                    self.winner_ant = ant

            self.updatePheromoneMatrix()

            if self.firstInit:
                self.firstInit = False

            self.setAnts(self.start_stop)
            # self.updated_pheromoneMatrix = [[0.0] * self.tour_size for _ in range(self.tour_size)]
            for col in self.df_updated_pheromoneMatrix.columns:
                self.df_updated_pheromoneMatrix[col].values[:] = 0.0
            print('{0}/{1} Searching...'.format(iteration + 1, self.iterations))

        self.updateProbabilityMatrix()

        #save_memory_df_to_local('./probabilityMatrixByAnts.pkl', self.ant_probability_Matrix)

        return self.shortest_distance, self.shortest_path, self.ant_probability_Matrix
