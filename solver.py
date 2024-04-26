import numpy as np
from simsnn.core.networks import Network

class MSTSolver():
    """MSTSolver

    Solve the Minimum Spanning Tree problem using the SNN algorithm

    Parameters
    ----------
    weight_matrix : np.ndarray
        Weight matrix of the graph, where weight_matrix[i,j] is the weight of the edge between vertices i and j
    """
    def __init__(self, weight_matrix) -> None:
        self.weight_matrix = weight_matrix
        self.weight_matrix = np.triu(self.weight_matrix, k=1) # Ensure the weight matrix is upper triangular
        self.edges = np.transpose(np.nonzero(self.weight_matrix)) # List of edges in the graph

        self.num_vertices = self.weight_matrix.shape[0]
        self.num_edges = self.edges.shape[0]

        self.max_edge_weight_count = int(np.unique(weight_matrix, return_counts=True)[1][1]) # Maximum number of times the same weight occurs
        self.max_edge_weight = (int(np.max(weight_matrix)) + 1) * self.max_edge_weight_count - 1 # Maximum weight of an edge after making them unique

        self.network = Network()
        self.init_algorithm_neurons()
        self.init_graph_neurons()
        self.init_graph_synapses()


    def init_algorithm_neurons(self) -> Network:
        """Initialise the neurons required for the algorithm"""

        # Create start neuron for starting each cycle
        self.start_neuron = self.network.createLIF(m=0, thr=1, ID='Start')

        # Create train to start the algorithm
        self.start_train = self.network.createInputTrain([1], loop=False, ID='1')
        self.network.createSynapse(self.start_train, self.start_neuron, w=1, d=1)

        # Create accept neuron for ending the algorithm
        self.accept_neuron = self.network.createLIF(m=1, thr=self.num_vertices-1, ID='Acc')

        # Create a reset neuron to reset the network before a new cycle
        self.reset_neuron = self.network.createLIF(m=0, thr=1, ID='Reset')
        self.network.createSynapse(self.reset_neuron, self.start_neuron, w=1, d=1)

        # Create inhibit neuron for inhibiting the neurons for a particular cycle
        self.inhibit_neuron = self.network.createLIF(m=1, thr=self.num_vertices, ID='Inhibit')
        self.network.createSynapse(self.reset_neuron, self.inhibit_neuron, w=-self.num_vertices, d=1)
        self.network.createSynapse(self.start_neuron, self.inhibit_neuron, w=self.num_vertices-1, d=1)
        self.network.createSynapse(self.inhibit_neuron, self.accept_neuron, w=1, d=1)

        # Add delay of maximum weight to ensure all spikes have propagated
        self.network.createSynapse(self.inhibit_neuron, self.reset_neuron, w=1, d=max(4, self.max_edge_weight+1))


    def init_graph_neurons(self) -> None:
        """Initialise the neurons for the vertices in the graph"""

        # Create a neuron for every vertex in the graph
        self.graph_neurons = []
        for i in range(self.num_vertices):
            self.init_graph_neuron(i)


    def init_graph_neuron(self, index) -> None:
        """Initialise the neurons for a vertex in the graph"""

        # Create graph neuron
        graph_neuron = self.network.createLIF(m=1, thr=self.num_vertices, ID=str(index)+'_graph')

        # At the start of a cycle, reset the graph neurons
        self.network.createSynapse(self.reset_neuron, graph_neuron, w=-self.num_vertices, d=1)
        self.network.createSynapse(self.start_neuron, graph_neuron, w=self.num_vertices-1, d=1)

        # Create part neuron for keeping track if the neuron is already part of the MST
        part_neuron = self.network.createLIF(m=1, V_init=(0 if index else self.num_vertices), V_reset=self.num_vertices, thr=self.num_vertices+1, ID=str(index)+'_part')

        # Create connections such that the part neurons all spike at the same time
        self.network.createSynapse(self.start_neuron, part_neuron, w=1, d=1)
        self.network.createSynapse(part_neuron, graph_neuron, w=1, d=1)

        # Create connection that prevents the inhibit neuron from spiking for a part neuron
        self.network.createSynapse(part_neuron, self.inhibit_neuron, w=-1, d=2)

        # Create neuron that adds the graph neuron to the MST
        add_neuron = self.network.createLIF(m=1, thr=2, ID=str(index)+'_add')
        self.network.createSynapse(graph_neuron, add_neuron, w=1, d=1)
        self.network.createSynapse(add_neuron, part_neuron, w=self.num_vertices+1, d=1)
        self.network.createSynapse(part_neuron, add_neuron, w=-1, d=1) # Prevent part neuron from adding again
        self.network.createSynapse(self.reset_neuron, add_neuron, w=-2, d=1) # Make sure the starting potential in each cycle is 1
        self.network.createSynapse(self.start_neuron, add_neuron, w=1, d=1)

        # If a neuron is added to the MST, inhibit all other graph neurons from spiking
        self.network.createSynapse(self.inhibit_neuron, graph_neuron, w=-self.num_vertices, d=1)
        self.network.createSynapse(graph_neuron, self.inhibit_neuron, w=1, d=1)
        self.network.createSynapse(self.inhibit_neuron, add_neuron, w=-2, d=1)

        self.graph_neurons.append((graph_neuron, part_neuron, add_neuron))


    def init_graph_synapses(self) -> None:
        """Initialise the synapses between the graph neurons that represent the edges in the graph"""

        self.edge_delays = {} # Dictionary to map unique edge delays to the corresponding edge
        weight_counts = {} # Dictionary to keep track of the number of times each weight has occurred
        
        # Create synapses between graph neurons for each edge
        for edge in self.edges:
            weight = self.weight_matrix[edge[0], edge[1]]

            if weight in weight_counts:
                weight_counts[weight] += 1
            else:
                weight_counts[weight] = 1

            delay = int(weight) * self.max_edge_weight_count + weight_counts[weight] - 1 # Ensure the delays are unique for each edge
            self.edge_delays[delay] = edge

            self.network.createSynapse(self.graph_neurons[edge[0]][0], self.graph_neurons[edge[1]][0], w=1, d=delay)
            self.network.createSynapse(self.graph_neurons[edge[1]][0], self.graph_neurons[edge[0]][0], w=1, d=delay)


    def solve(self) -> np.ndarray:
        """Solve the Minimum Spanning Tree problem using the SNN algorithm"""

        time = 0 # Variable to keep track of the passed time steps since the start of the current cycle
        time_added = np.zeros(self.num_vertices) # Array to keep track of the time each vertex is added to the MST
        
        # Run the simulation of the SNN
        while True:
            self.network.step() # Run a simulation step

            if self.start_neuron.out: # Reset the time variable at the start of each cycle
                time = 0

            for i in range(self.num_vertices):
                if self.graph_neurons[i][2].out: # Record the spike times of the add neurons
                    time_added[i] = time
                    break

            if self.accept_neuron.out: # Break the loop if the algorithm has finished
                break

            time += 1

        time_added = time_added - 3 # Subtract the delays from the algorithm synapses

        # Construct the MST using the recorded spike times
        mst = np.zeros_like(self.weight_matrix)
        for i in range(1, self.num_vertices):
            edge = self.edge_delays[int(time_added[i])]
            mst[edge[0], edge[1]] = self.weight_matrix[edge[0], edge[1]]

        return mst
