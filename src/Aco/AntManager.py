import pandas as pd

from src.Aco.Ant import Ant


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
        self.nodes_amount = len(stops)
        self.start_stop = 0 if start_stop is None else start_stop

        self.firstInit = True

        # Initializing Pheromone
        self.tour_size = len(stops)
        # self.df_pheromoneMatrix = pd.read_pickle('data/model')
        self.df_pheromoneMatrix = []
        self.df_updated_pheromoneMatrix = []
        self.load_df_pheromoneMatrix()
        self.setup_df_updated_pheromoneMatrix()
        #self.pheromoneMatrix = [
        #    [0.0] * self.tour_size for _ in range(self.tour_size)]
        #self.updated_pheromoneMatrix = [
        #    [0.0] * self.tour_size for _ in range(self.tour_size)]

        self.discountAlpha = float(discountAlpha)
        self.discountBeta = float(discountBeta)
        self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations

        # Initializing the List of Ants
        self.antCount = vehicleCount
        self.antWeight = vehicleWeight
        self.antVolume = vehicleVolume
        self.ants = self.setAnts(self.start_stop)

        # For Solution
        self.shortest_distance = None
        self.shortest_path = None

    def setAnts(self, startStop):
        print("-Setting up Ants-")
        if self.firstInit:
            return [Ant(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromoneMatrix,
                        self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, firstInit=True)
                    for _ in range(self.antCount)]

        for ant in self.ants:
            ant.__init__(startStop, startStop, self.antWeight, self.antVolume, self.nodes, self.df_pheromoneMatrix,
                         self.discountAlpha, self.discountBeta, self.pheromone_evaporation_coefficient, firstInit=False)

    def setup_df_updated_pheromoneMatrix(self):
        df_new_pheromoneMatrix = pd.DataFrame(index=[node.hashIdentifier for node in self.nodes],
                                              columns=[node.hashIdentifier for node in self.nodes])
        df_new_pheromoneMatrix.fillna(value=0.0, inplace=True)
        self.df_updated_pheromoneMatrix = df_new_pheromoneMatrix

    def load_df_pheromoneMatrix(self):
        try:
            df_pickle = pd.read_pickle('./pheromoneMatrix.pkl')
            if df_pickle.empty:
                df_new_pickle = pd.DataFrame(index=[node.hashIdentifier for node in self.nodes],
                                             columns=[node.hashIdentifier for node in self.nodes])
                df_new_pickle.fillna(value=0.0, inplace=True)
                df_new_pickle.to_pickle('./pheromoneMatrix.pkl')
                self.df_pheromoneMatrix = df_new_pickle
            else:
                for node in self.nodes:
                    if node.hashIdentifier not in df_pickle.index:
                        df_pickle[node.hashIdentifier] = df_pickle.index
                    else:
                        print("vorhanden")
                df_pickle.fillna(value=0.0, inplace=True)
                df_pickle.to_pickle('./pheromoneMatrix.pkl')
                self.df_pheromoneMatrix = df_pickle
        except:
            print("-No Pickle file to load from-")
            print("-Creating new Pickle file-")
            df_new_pickle = pd.DataFrame(index=[node.hashIdentifier for node in self.nodes],
                                         columns=[node.hashIdentifier for node in self.nodes])
            df_new_pickle.fillna(value=0.0, inplace=True)
            df_new_pickle.to_pickle('./pheromoneMatrix.pkl')
            self.df_pheromoneMatrix = df_new_pickle

    def updateDFPheromoneMatrix(self):
        for df_start in self.df_pheromoneMatrix:
            for df_end in self.df_pheromoneMatrix:
                self.df_pheromoneMatrix.at[df_start, df_end] *= (1 - self.pheromone_evaporation_coefficient)
                self.df_pheromoneMatrix.at[df_start, df_end] += self.df_updated_pheromoneMatrix.at[df_start, df_end]

    def updatePheromoneMatrix(self):
        for start in range(len(self.pheromoneMatrix)):
            for end in range(len(self.pheromoneMatrix)):
                self.pheromoneMatrix[start][end] *= (1 - self.pheromone_evaporation_coefficient)
                self.pheromoneMatrix[start][end] += self.updated_pheromoneMatrix[start][end]

    def updatePheromoneMatrixByAnt(self, ant):
        tour = ant.getTour()
        for index, stop in enumerate(tour):
            stop_b = 0
            if index == len(tour) - 1:
                break
                # stop_b = tour[0]
            else:
                stop_b = tour[index + 1]

            new_pheromoneValue = self.pheromone_constant / ant.get_travelled_Distance()  # 1/Total Length
            self.df_updated_pheromoneMatrix.at[stop.hashIdentifier, stop_b.hashIdentifier] += new_pheromoneValue
            self.df_updated_pheromoneMatrix.at[stop_b.hashIdentifier, stop.hashIdentifier] += new_pheromoneValue
            #self.updated_pheromoneMatrix[stop.stopid][stop_b.stopid] += new_pheromoneValue
            #self.updated_pheromoneMatrix[stop_b.stopid][stop.stopid] += new_pheromoneValue

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
                    self.shortest_path = ant.getTour()
                if ant.get_travelled_Distance() < self.shortest_distance:
                    self.shortest_distance = ant.get_travelled_Distance()
                    self.shortest_path = ant.getTour()

            #self.updatePheromoneMatrix()
            self.updateDFPheromoneMatrix()

            if self.firstInit:
                self.firstInit = False

            self.setAnts(self.start_stop)
            #self.updated_pheromoneMatrix = [[0.0] * self.tour_size for _ in range(self.tour_size)]
            for col in self.df_updated_pheromoneMatrix.columns:
                self.df_updated_pheromoneMatrix[col].values[:] = 0.0
            # if (iteration + 1) % 50 == 0:
            #    print('{0}/{1} Searching...'.format(iteration + 1, self.iterations))
            print('{0}/{1} Searching...'.format(iteration + 1, self.iterations))
            # if _ % 10 == 0:
            #     print(f"Iteration: {_} Best Fitness: {self.shortest_distance}")

        self.df_pheromoneMatrix.to_pickle('./pheromoneMatrix.pkl')
        return self.shortest_distance, self.shortest_path, self.pheromoneMatrix, self.df_pheromoneMatrix
