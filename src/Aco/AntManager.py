from src.Aco.Ant import Ant


class AntManager:

    def __init__(self,
                 stops,
                 start_stop,
                 vehicleCount,
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
        self.pheromoneMatrix = [
            [0.0] * self.tour_size for _ in range(self.tour_size)]
        self.updated_pheromoneMatrix = [
            [0.0] * self.tour_size for _ in range(self.tour_size)]

        self.discountAlpha = float(discountAlpha)
        self.discountBeta = float(discountBeta)
        # self.pheromone_evaporation_coefficient = pheromone_evaporation_coefficient
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations

        # Initializing the List of Ants
        self.antCount = vehicleCount
        self.ants = self.setAnts(self.start_stop)

        # For Solution
        self.shortest_distance = None
        self.shortest_path = None

    def setAnts(self, startStop):
        if self.firstInit:
            return [Ant(startStop, startStop, self.nodes, self.pheromoneMatrix,
                        self.discountAlpha, self.discountBeta, firstInit=True) for _ in range(self.antCount)]

        for ant in self.ants:
            ant.__init__(startStop, startStop, self.nodes, self.pheromoneMatrix, self.discountAlpha, self.discountBeta)

    def updatePheromoneMatrix(self):
        for start in range(len(self.pheromoneMatrix)):
            for end in range(len(self.pheromoneMatrix)):
                # self.pheromoneMatrix[start][end] *= (1 - self.pheromone_evaporation_coefficient)
                self.pheromoneMatrix[start][end] += self.updated_pheromoneMatrix[start][end]

    def updatePheromoneMatrixByAnt(self, ant):
        tour = ant.getTour()
        for index, stop in enumerate(tour):
            stop_b = 0
            if index == len(tour) - 1:
                stop_b = tour[0]
            else:
                stop_b = tour[index + 1]

            new_pheromoneValue = self.pheromone_constant / ant.get_travelled_Distance()
            self.updated_pheromoneMatrix[stop.stopid][stop_b.stopid] += new_pheromoneValue
            self.updated_pheromoneMatrix[stop_b.stopid][stop.stopid] += new_pheromoneValue

    def mainloop(self):
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

            self.updatePheromoneMatrix()
            self.setAnts(self.start_stop)
            self.updated_pheromoneMatrix = [[0.0] * self.tour_size for _ in range(self.tour_size)]
            if (iteration + 1) % 50 == 0:
                print('{0}/{1} Searching...'.format(iteration + 1, self.iterations))

            # if _ % 10 == 0:
            #     print(f"Iteration: {_} Best Fitness: {self.shortest_distance}")

        return self.shortest_distance, self.shortest_path, self.pheromoneMatrix, self.updated_pheromoneMatrix
