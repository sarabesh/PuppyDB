import numpy as np
import heapq
from collections import Counter

class Search:
    def search(self, query_vector, k):
        raise NotImplementedError

class KnnSearch(Search):
    def __init__(self, puppydb):
        self.puppydb = puppydb
    
    def index(self):
        # get all vectors from the vector store
        self.vectors = self.puppydb.get_all_vectors()
       
    def search(self, query_vector, k=5):
        # Exact KNN logic here
        # compute distances to query_vector
        # return the k nearest vectors
        if not self.vectors:
            return []
        # Extract vector data and compute distances
        similarities = []
        print(f"query_vector: {query_vector.shape}")
        
        for vector_id, vector in self.vectors:
            #cosine similarity
            similarity = np.dot(vector, query_vector) / (np.linalg.norm(vector) * np.linalg.norm(query_vector))
            heapq.heappush(similarities, (-similarity, vector_id))  # Use a min-heap to keep track of top k
        
        # find metadata for the top k vectors
        results = []
        for i in range(min(k, len(similarities))):
            # pop the smallest similarity (which is actually the largest due to negation)
            similarity, vector_id = heapq.heappop(similarities)
            similarity = -similarity
            # retrieve the vector and metadata
            vector, metadata = self.puppydb.get_vector(vector_id)
            results.append((vector_id, similarity , metadata))
        
        return results
        

class Node:
    def __init__(self, vector_id, vector, layer_id, next_node=None):
        self.vector_id = vector_id
        self.vector = vector
        self.layer_id = layer_id  # Layer ID this node belongs to
        self.neighbors = []  # List of neighboring nodes
        self.next_node = next_node  # Pointer to the next node in the next layer
    def __repr__(self):
        return f"{self.layer_id}-{self.vector_id} --->  {self.next_node}"

        
# reference: https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW
#list of layers of graphs
class HNSWSearch(Search):
    def __init__(self, puppydb):
        self.puppydb = puppydb
        
        self.max_layer = 2  # Maximum number of layers in the HNSW graph, starting from 0 so 0,1,2
        self.layers = [[] for _ in range(self.max_layer + 1)]
        self.collection = []  
        self.max_neighbors_per_layer = 5 # Maximum number of neighbors per node per layer, this is a hyperparameter

    def insert_graph(self, vector_id, vector, layer_id, next_node=None):
        # Insert a node into the graph at the specified layer
        # create a new node with vector_id, vector, layer_id and next_node, todo: add neighbors
        node = Node(vector_id, vector, layer_id, next_node)
        layer = self.layers[layer_id]  
        layer.append(node) # currently appending to the layer, byut this should be a graph structure
        return node

    # Insert a vector into the HNSW
    def insert(self, vector_id, vector):
        # more nodes in lower layers, less in upper layers
        #  To achieve this, we use exponential sampling np.random.exponential gives vals between 0 and inf, we take min of that and max_layer
        # P(l)=e ^(-l/scale) where l is the layer and scale is a hyperparameter that controls the distribution of nodes across layers
        scale = 1/np.log(self.max_neighbors_per_layer)
        insert_layer = min(int(np.random.exponential(scale)), self.max_layer ) #gives frequency of nodes in each layer, more nodes in lower layers, less in upper layers
        
        # Insert the node into the graph at the until specfied layer, and link it to the next node in the next layer
        prev_node = None
        for i in range(insert_layer,-1,-1):
            current_node = self.insert_graph(vector_id, vector, i)
            if prev_node is not None:
                # If we have a previous node, link it to the current node
                prev_node.next_node =  current_node
            prev_node = current_node  # The next node for the next layer will be this one
        pass

    def index(self):
        # HNSW indexing logic here
        # This would typically involve building a graph structure
        # For simplicity, we will not implement the actual HNSW algorithm here
        #load index from disk/build index if not exists
        #build index
        #build layers of graph
        vectors = self.puppydb.get_all_vectors()
        if not vectors:
            return []
    
        for vector_id, vector in vectors:
            self.insert(vector_id, vector)
        

        for layer in reversed(self.layers):
            print(f"Layer {self.layers.index(layer)}: {len(layer)} nodes")
            print([node for node in layer])

        # for layer in self.layers:
        #     print(f"Layer {self.layers.index(layer)}: {len(layer)} nodes")
        #     print("Node IDs:", [node.vector_id for node in layer])
        # self.layers would be a list of Node objects representing the HNSW graph layers
       

    def search(self, query_vector, k=5):
        # HNSW
        # Extract vector data and compute distances
        similarities = []
        print(f"query_vector: {query_vector.shape}")
        #self.layers from index() would be used to search
        pass
        # For simplicity, we will not implement the actual HNSW algorithm here