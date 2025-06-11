import numpy as np
import heapq
from bisect import insort
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
    def __init__(self, vector_id, vector, layer_id):
        self.vector_id = vector_id
        self.vector = vector
        self.layer_id = layer_id  # Layer ID this node belongs to
        self.neighbors = []  # List of neighboring nodes
    def __repr__(self):
        neighbor_info = [f"{n.layer_id}-{n.vector_id}" for n in self.neighbors]
        return f"{self.layer_id}-{self.vector_id} --> neighbors: {neighbor_info}"

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cosine_distance(vec1, vec2):
    """Compute cosine distance between two vectors."""
    return 1 - cosine_similarity(vec1, vec2)

# reference: https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW
#list of layers of graphs
class HNSWSearch(Search):
    def __init__(self, puppydb):
        self.puppydb = puppydb
        
        self.max_layer = 2  # Maximum number of layers in the HNSW graph, starting from 0 so 0,1,2
        self.layers = [{} for _ in range(self.max_layer + 1)]
        self.collection = []  
        self.max_neighbors_per_layer = 15 # Maximum number of neighbors per node per layer, this is a hyperparameter
        self.entry_point = None
    
    def __repr__(self):
        return f"HNSWSearch(max_layer={self.max_layer}, max_neighbors_per_layer={self.max_neighbors_per_layer}, entry_point={self.entry_point})"

    def print_layers(self):
    # print("=== HNSW Layers ===")
        for i in reversed(range(len(self.layers))):  # Print from top layer down
            layer = self.layers[i]
            print(f"\nLayer {i} ({len(layer)} nodes):")
            for vector_id, node in layer.items():
                print(f"  - {node}") 

    def _search_layer_neighbors(self, layer, entry_node, query_vector, ef=5):
        # print(f"Searching neighbors in layer {entry_node.layer_id} from entry node {entry_node}")
        # # Search for neighbors in the specified layer
        #  find ef nearest neighbors in the layer
        entry_dist = cosine_distance(entry_node.vector, query_vector)
        best = (entry_dist, entry_node)

        nearest_neighbors = [best] # contains the best neighbors found so far
        candidates = [best] # contains the candidates to be explored
        heapq.heapify(candidates) #min-heap to keep track of the best candidates

        visited = set()  # Keep track of visited nodes to avoid cycles
        visited.add(entry_node.vector_id)

        while candidates:
            current_distance, current_node = heapq.heappop(candidates)  # Get the current best candidate

            # if current_distance is worse than the worst in nearest_neighbors, we can skip current_node
            # nearest_neightbors is sorted by distance, so the last element is the worst
            if nearest_neighbors[-1][0] < current_distance:
                break

            # explore neighbors of the current node
            for neighbor in current_node.neighbors:
                if neighbor.vector_id not in visited:
                    visited.add(neighbor.vector_id)

                    neighbor_distance = cosine_distance(neighbor.vector, query_vector)

                    # If the neighbor is better than the worst in nearest_neighbors, add it to candidates and nearest_neighbors(remove worst one if lenght exceeds ef)
                    if len(nearest_neighbors) < ef or neighbor_distance < nearest_neighbors[-1][0]:
                      
                        heapq.heappush(candidates, (neighbor_distance, neighbor))
                        insort(nearest_neighbors, (neighbor_distance, neighbor))  # Insert and sort by distance
                       
                        # If we have more than ef neighbors, remove the worst one
                        if len(nearest_neighbors) > ef:
                            nearest_neighbors.pop()
        # print(nearest_neighbors)
        return nearest_neighbors
    

    # Insert a vector into the HNSW
    def insert(self, vector_id, vector):
    
        print(f"Inserting vector {vector_id} into HNSW graph")
        #Bootstrap the graph if it is empty
        # If no nodes in graph, create node in all layers
        if not self.layers[0]:
            i = None
            prev_node = None
            for i in range(len(self.layers)-1,-1,-1):
                newnode = Node(vector_id, vector, i)
                self.layers[i][vector_id] = newnode # Insert into the first layer
                if i == len(self.layers) - 1:
                    self.entry_point = newnode
            print(f"entry point:{self.entry_point}")
            return


        # more nodes in lower layers, less in upper layers
        #  To achieve this, we use exponential sampling np.random.exponential gives vals between 0 and inf, we take min of that and max_layer
        # P(l)=e ^(-l/scale) where l is the layer and scale is a hyperparameter that controls the distribution of nodes across layers
        scale = 1/np.log(self.max_neighbors_per_layer)
        insert_layer = min(int(np.random.exponential(1)), self.max_layer ) 
        

        # Greedy search from top layer down to insert_layer + 1, just to find entry point at insert_layer
        current_node = self.entry_point #at top layer
        for i in range(self.max_layer, insert_layer, -1):
            nearest_neighbors = self._search_layer_neighbors(self.layers[i], current_node, vector, ef=1)
            print(f"Layer {i} nearest neighbors: {nearest_neighbors[0][1].vector_id, nearest_neighbors[0][1].layer_id} with distance {nearest_neighbors[0][0]}")
            current_node = self.layers[i-1][nearest_neighbors[0][1].vector_id]  # Best neighbor becomes new entry point for lower layer

        print(f"insert_layer: {insert_layer}, current_node: {current_node.vector_id}, current_node.layer_id: {current_node.layer_id}")

        # Insert node into layers from insert_layer to 0
        for i in range(insert_layer, -1, -1):
            
            if i!=insert_layer:
                # For next layer down, the new node becomes current_node
                current_node = self.layers[i][nearest_neighbors[0][1].vector_id]  # Best neighbor becomes new entry point for lower layer
                
            print(f"Getting neighbors in layer {i} from current_node {current_node.layer_id}-{current_node.vector_id}")
            nearest_neighbors = self._search_layer_neighbors(self.layers[i], current_node, vector, ef=10)
            print(f"Layer {i} nearest neighbors: {[f'{n[1].layer_id}-{n[1].vector_id}' for n in nearest_neighbors]}")
            
            
                # Create new node for the current layer
            newnode = Node(vector_id, vector, i)
                # Insert new node into current layer
            self.layers[i][vector_id] = newnode
                


            # Connect bidirectional edges
            for _, neighbor in nearest_neighbors[:self.max_neighbors_per_layer]:
                newnode.neighbors.append(neighbor)
                neighbor.neighbors.append(newnode)

                # Optional pruning
                if len(neighbor.neighbors) > self.max_neighbors_per_layer:
                    self._prune_neighbors(neighbor)
                    pass # Pruning logic can be added here if needed

          

    def _prune_neighbors(self, node):
        """Prune neighbors of a node to maintain max_neighbors_per_layer."""
         # First sort neighbors by distance
        node.neighbors.sort(key=lambda n: cosine_distance(node.vector, n.vector))
        while len(node.neighbors) > self.max_neighbors_per_layer:
            node.neighbors.pop() # remove the farthest neighbor
        return 


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
            self.print_layers()
            self.insert(vector_id, vector)

        

        for layer in reversed(self.layers):
            print(f"Layer {self.layers.index(layer)}: {len(layer)} nodes")
            print([node for node in layer])
       

    def search(self, query_vector, k=5):
        # HNSW

        if self.entry_point is None:
            print("Empty graph!")
            return []
    
        current_node = self.entry_point
        for i in range(self.max_layer, 0, -1):
            # Greedy search (ef=1) in layer i
            nearest_neighbors = self._search_layer_neighbors(self.layers[i], current_node, query_vector, ef=1)
            current_node = self.layers[i - 1][nearest_neighbors[0][1].vector_id]  # Move to next lower layer

        # Full EF search in layer 0
        ef_search = max(k * 2, 10)  # Larger ef for better recall
        nearest_neighbors = self._search_layer_neighbors(self.layers[0], current_node, query_vector, ef=ef_search)

        # Return top-k results as (vector_id, distance)
        top_k = nearest_neighbors[:k]
        
         # find metadata for the top k vectors
        results = []
        for i in range(min(k, len(top_k))):
            # pop the smallest similarity (which is actually the largest due to negation)
            similarity, node = top_k[i]
            vector_id = node.vector_id
            # similarity = -similarity
            # retrieve the vector and metadata
            _, metadata = self.puppydb.get_vector(vector_id)
            results.append((vector_id, similarity , metadata))
        
        return results

    
    
       

       