#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03.02.2024

<This software implements spike sorting and detection a la Quiroga: http://www.scholarpedia.org/article/Spike_sorting>
It is based on the MCS Tutorial: https://mcspydatatools.readthedocs.io/en/latest/McsPy-Tutorial_DataAnalysis.html

Copyright (C) <2024>  <Freya Ebba Christ>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: Freya Ebba Christ
"""

import numpy as np
import pandas as pd
import random
import heapq
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt

class UnionFind:
    """A class to perform union find operations."""
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, vertex):
        """Finds the root of the vertex."""
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, root1, root2):
        """Unites two subsets into a single subset."""
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        elif self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

def kruskals_algorithm(edges, num_nodes):
    """Implements Kruskal's algorithm to find the Minimum Spanning Tree (MST)."""
    uf = UnionFind(num_nodes)
    mst = []
    for edge in sorted(edges, key=lambda e: e[2]):
        u, v, weight = edge
        if uf.find(u) != uf.find(v):
            uf.union(uf.find(u), uf.find(v))
            mst.append(edge)
    return mst

def make_graph_eulerian(mst, num_nodes):
    """Adds minimum number of edges to make the graph Eulerian if necessary."""
    degree = [0] * num_nodes
    for u, v, _ in mst:
        degree[u] += 1
        degree[v] += 1
    odd_degree_nodes = [i for i, deg in enumerate(degree) if deg % 2 != 0]
    while odd_degree_nodes:
        u = odd_degree_nodes.pop()
        v = odd_degree_nodes.pop(0)
        mst.append((u, v, 1))  # Adding an edge with a placeholder weight
    return mst

def eulerian_mst_to_tsp_route(eulerian_mst, num_cities):
    """Converts an Eulerian MST to a TSP route by shortcutting."""
    adj_list = {i: [] for i in range(num_cities)}
    for u, v, _ in eulerian_mst:
        adj_list[u].append(v)
        adj_list[v].append(u)
    # Find Eulerian circuit
    eulerian_circuit = find_eulerian_circuit(adj_list)
    # Shortcutting: Remove repeated visits to form a Hamiltonian cycle
    tsp_route = shortcut_eulerian_to_hamiltonian(eulerian_circuit)
    return tsp_route

def find_eulerian_circuit(adj_list):
    """Finds an Eulerian circuit in the given adjacency list."""
    start_vertex = next((v for v, edges in adj_list.items() if edges), None)
    if start_vertex is None:
        return []
    stack = [start_vertex]
    path = []
    while stack:
        vertex = stack[-1]
        if adj_list[vertex]:
            next_vertex = adj_list[vertex].pop()
            adj_list[next_vertex].remove(vertex)
            stack.append(next_vertex)
        else:
            path.append(stack.pop())
    return path[::-1]

def shortcut_eulerian_to_hamiltonian(eulerian_circuit):
    """Removes repeated nodes to convert an Eulerian circuit to a Hamiltonian cycle."""
    visited = set()
    hamiltonian_cycle = []
    for node in eulerian_circuit:
        if node not in visited:
            visited.add(node)
            hamiltonian_cycle.append(node)
    hamiltonian_cycle.append(hamiltonian_cycle[0])  # Adding the start node at the end to complete the cycle
    return hamiltonian_cycle

def a_star_optimize(route, distance_matrix):
    """Applies A* algorithm to optimize a given route."""
    num_cities = len(route)
    start_node = route[0]
    goal_node = route[-1]
    heap = [(0, start_node, tuple(route))]
    visited = set()
    
    while heap:
        _, current_node, current_path = heapq.heappop(heap)
        if current_node == goal_node:
            return list(current_path)
        if (current_node, current_path) in visited:
            continue
        visited.add((current_node, current_path))
        for neighbor in range(num_cities):
            if neighbor != current_node and neighbor not in current_path:
                new_path = list(current_path)
                new_path = new_path[:new_path.index(current_node) + 1] + [neighbor] + new_path[new_path.index(current_node) + 1:]
                cost = calculate_route_length(new_path, distance_matrix)
                heapq.heappush(heap, (cost, neighbor, tuple(new_path)))
    return route

def rank_routes(population, distance_matrix):
    fitness_results = {i: calculate_route_length(route, distance_matrix) for i, route in enumerate(population)}
    return sorted(fitness_results.items(), key=lambda x: x[1])

def selection(ranked_routes, elite_size):
    if len(ranked_routes) <= elite_size:
        return [i for i, _ in ranked_routes]
    else:
        return [index for index, _ in ranked_routes[:elite_size]]

def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]

def breed(parent1, parent2):
    child = []
    geneA, geneB = sorted(random.sample(range(len(parent1)), 2))
    childP1 = parent1[geneA:geneB]
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def cx_ordered_crossover(parent1, parent2):
    """Order Crossover (OX)"""
    size = min(len(parent1), len(parent2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a
    mapping = {gene: None for gene in parent1[a:b]}
    child = [None] * len(parent1)
    child[a:b] = parent1[a:b]
    index = b
    for gene in parent2[b:] + parent2[:b]:
        if gene not in mapping:
            child[index] = gene
            index = (index + 1) % len(parent1)
    return child

def cx_cycle_crossover(parent1, parent2):
    """Cycle Crossover (CX)"""
    size = min(len(parent1), len(parent2))
    mapping = {gene: i for i, gene in enumerate(parent2)}
    child = [None] * len(parent1)
    index = 0
    while True:
        child[index] = parent1[index]
        index = mapping[parent1[index]]
        if index == 0:
            break
    for i, gene in enumerate(child):
        if gene is None:
            child[i] = parent2[i]
    return child

def mutate_swap(individual, mutation_rate):
    """Swap Mutation"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def mutate_insert(individual, mutation_rate):
    """Insert Mutation"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            if i != j:
                individual.insert(j, individual.pop(i))
    return individual

def mutate_reverse_sequence(individual, mutation_rate):
    """Reverse Sequence Mutation"""
    if random.random() < mutation_rate:
        a, b = sorted(random.sample(range(len(individual)), 2))
        individual[a:b] = reversed(individual[a:b])
    return individual

def mutate_population(population, mutation_rate):
    return [mutate_swap(individual, mutation_rate) for individual in population]

def opt_3_exchange(route, distance_matrix):
    """Performs an Opt-3 exchange operation to optimize the route."""
    best_distance = calculate_route_length(route, distance_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                for k in range(j + 2, len(route)):
                    new_route = route[:i+1] + route[j:k] + route[i+1:j] + route[k:]
                    new_distance = calculate_route_length(new_route, distance_matrix)
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
    return route


def next_generation(current_gen, elite_size, mutation_rate, crossover_rate, distance_matrix):
    ranked_routes = rank_routes(current_gen, distance_matrix)
    selection_results = selection(ranked_routes, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size, crossover_rate)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def calculate_route_length(route, distance_matrix):
    # Ensure route is not None and is a list or similar iterable
    if route is None or not hasattr(route, '__iter__') or len(route) == 0:
        print("Route:", route)
        print("Route type:", type(route))
        raise ValueError("Route is empty or None.")
        
    # Convert route to a NumPy array for advanced indexing
    route_np = np.array(route)
    
    # Check if route_np is None or empty
    if route_np is None or len(route_np) == 0:
        raise ValueError("Route is empty or None.")
    
    # Check if distance_matrix is a 2D array
    if len(distance_matrix.shape) != 2:
        raise ValueError("distance_matrix must be a 2D NumPy array.")
    
    # Use advanced indexing to select the distances between consecutive cities
    # and sum them up. Also include the return to the start for a complete cycle.
    return np.sum(distance_matrix[route_np[:-1], route_np[1:]]) + distance_matrix[route_np[-1], route_np[0]]

def generate_eulerian_mst_based_population(pop_size, num_cities, edges):
    population = []
    for _ in range(pop_size):
        mst = kruskals_algorithm(edges, num_cities)
        if not mst:
            print("Error: Failed to generate MST")
            continue
        
        eulerian_mst = make_graph_eulerian(mst, num_cities)
        if not eulerian_mst:
            print("Error: Failed to convert MST to Eulerian circuit")
            continue
        
        tsp_route = eulerian_mst_to_tsp_route(eulerian_mst, num_cities)
        
        # Ensure the generated route does not contain out-of-bounds indices
        tsp_route = tsp_route[:num_cities]
        
        if not tsp_route:
            print("Error: Failed to convert Eulerian circuit to TSP route")
            continue
        
        population.append(tsp_route)
    return population


def breed_population(matingpool, elite_size, crossover_rate):
    children = matingpool[:elite_size]  # Carry forward elite routes unchanged
    pool = random.sample(matingpool, len(matingpool))
    for i in range(elite_size, len(matingpool)):
        if random.random() < crossover_rate:  # Dynamic crossover rate
            child = cx_ordered_crossover(pool[i], pool[len(matingpool)-i-1])
        else:
            child = cx_cycle_crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def create_distance_matrix(num_cities):
    matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix

def adaptive_termination_criteria(generations, current_best_distance, previous_best_distance, threshold=10):
    """Checks for improvement in the best distance and adapts termination criteria."""
    if generations == 0:
        return False  # Continue until at least one generation
    if current_best_distance < previous_best_distance:
        return False  # Continue if there's improvement
    else:
        return generations < threshold  # Terminate after threshold generations without improvement

def adaptive_fitness_threshold(initial_threshold, generations, current_best_distance, previous_best_distance):
    """Adjusts the fitness threshold based on the improvement in the best distance."""
    if generations == 0:
        return initial_threshold
    if current_best_distance < previous_best_distance:
        return initial_threshold  # Reset to initial threshold if there's improvement
    else:
        return initial_threshold * 1.1  # Increase threshold if there's no improvement

def two_opt_swap(route, i, j):
    """Performs a 2-opt swap by reversing the route segment between i and j."""
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route

def k_opt(route, distance_matrix, max_k):
    """Enhances the route by iteratively applying k-opt swaps."""
    improvement = True
    while improvement:
        improvement = False
        for k in range(2, max_k + 1):  # Dynamically choose k-opt depth
            for i in range(len(route) - (k + 1)):
                for j in range(i+k, len(route)-1):
                    new_route = two_opt_swap(route, i, j)
                    if calculate_route_length(new_route, distance_matrix) < calculate_route_length(route, distance_matrix):
                        route = new_route
                        improvement = True
                        break  # Break to restart the search with the improved route
                if improvement:
                    break
            if improvement:
                break
    return route

def lkh_inspired_optimization(route, distance_matrix, max_k):
    """Performs an LKH-inspired optimization on the given route using k-opt swaps."""
    # Given the implementation of k_opt, you can directly use it here.
    # The k_opt function already includes a loop for improvement,
    # so it can replace the previous for-loop structure for k and swapping.
    optimized_route = k_opt(route, distance_matrix, max_k)
    return optimized_route

def lkh_inspired_optimization_with_partitioning(route, distance_matrix, max_k, num_partitions):
    """Performs an LKH-inspired optimization with partitioning and segment management."""
    optimized_route = optimize_route_segments(route, distance_matrix, num_partitions, max_k)
    return optimized_route

def calculate_candidate_list(distance_matrix, candidate_list_size=5):
    """Calculates a candidate list for each node based on distance."""
    candidate_list = {}
    for i in range(len(distance_matrix)):
        distances = list(enumerate(distance_matrix[i]))
        distances.sort(key=lambda x: x[1])
        # Exclude the first element as it is the distance to itself (0)
        candidate_list[i] = [index for index, distance in distances[1:candidate_list_size+1]]
    return candidate_list

def two_opt_swap_with_candidates(route, i, j):
    """Performs a 2-opt swap, same as before."""
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route

def k_opt_with_candidates(route, distance_matrix, candidate_list, max_k):
    """Enhances the route by iteratively applying k-opt swaps using a dynamic candidate list."""
    improvement = True
    while improvement:
        improvement = False
        for k in range(2, max_k + 1):
            # Iterate through the route based on the candidate list
            for i in range(len(route) - (k + 1)):
                candidates = candidate_list[route[i]]
                for candidate in candidates:
                    j = route.index(candidate)
                    if j > i + 1 and j < len(route) - 1:  # Ensure valid swap
                        new_route = two_opt_swap_with_candidates(route, i, j)
                        if calculate_route_length(new_route, distance_matrix) < calculate_route_length(route, distance_matrix):
                            route = new_route
                            improvement = True
                            break  # Break to restart with the improved route
                if improvement:
                    break
            if improvement:
                break
    return route

def lkh_inspired_optimization_with_candidates(route, distance_matrix, max_k):
    """Incorporates candidate lists into the LKH-inspired optimization."""
    candidate_list = calculate_candidate_list(distance_matrix)
    optimized_route = k_opt_with_candidates(route, distance_matrix, candidate_list, max_k)
    return optimized_route

def merge_segments(segments):
    """Merges optimized segments back into a single route."""
    return [city for segment in segments for city in segment]

def partition_route(route, num_partitions):
    """Divides the route into smaller segments for independent optimization."""
    partition_size = len(route) // num_partitions
    partitions = [route[i:i + partition_size] for i in range(0, len(route), partition_size)]
    return partitions

def optimize_route_segments(route, distance_matrix, num_partitions, max_k):
    """Partition the route, optimize each segment, then merge and optimize again."""
    partitions = partition_route(route, num_partitions)
    optimized_partitions = [k_opt(partition, distance_matrix, max_k) for partition in partitions]
    merged_route = merge_segments(optimized_partitions)
    # Optional: Perform a final optimization pass on the merged route
    final_optimized_route = k_opt(merged_route, distance_matrix, max_k)
    return final_optimized_route


def collect_data(num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_threshold, max_k, num_partitions, final_distance, iteration):

    global data_collection_list  # Ensure we're modifying the global list
    global data_collection  # Reference the global DataFrame
    # Existing code ...
    for gen in range(generations):
        # Existing loop code...
        # Collect data after each iteration
        iteration_data = {
            'iteration': iteration,
            'num_cities': num_cities,
            'pop_size': pop_size,
            'elite_size': elite_size,
            'mutation_rate': mutation_rate,
            'generation': gen,
            'initial_crossover_rate': initial_crossover_rate,
            'final_crossover_rate': final_crossover_rate,
            'initial_threshold': initial_threshold,
            'max_k': max_k,
            'num_partitions': num_partitions,
            'final_distance': final_distance
        }
        data_collection_list.append(iteration_data)

# Define a DataFrame to hold the collected data
columns = ['iteration', 'num_cities', 'pop_size', 'elite_size', 'mutation_rate', 'generations', 'initial_crossover_rate', 'final_crossover_rate', 'initial_threshold', 'max_k', 'num_partitions', 'final_distance']
data_collection_list = []
iteration = 0

def genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_threshold, max_k, num_partitions):
    edges = [(i, j, np.random.randint(1, 100)) for i in range(num_cities) for j in range(i+1, num_cities)]
    pop = generate_eulerian_mst_based_population(pop_size, num_cities, edges)
    distance_matrix = create_distance_matrix(num_cities)
    best_distance = float('inf')
    fitness_threshold = initial_threshold
    best_route = None
    global iteration
    
    print("Initial distance: " + str(calculate_route_length(pop[0], distance_matrix)))
    for gen in range(generations):
        crossover_rate = initial_crossover_rate - ((initial_crossover_rate - final_crossover_rate) * gen / generations)
        pop = next_generation(pop, elite_size, mutation_rate, crossover_rate, distance_matrix)
        print("Population size after next_generation:", len(pop))  
        for idx in range(len(pop)):
            pop[idx] = lkh_inspired_optimization_with_partitioning(pop[idx], distance_matrix, max_k, num_partitions)
            print("Route length after LKH optimization:", calculate_route_length(pop[idx], distance_matrix))  
        
        if pop:
            current_best_distance = calculate_route_length(pop[0], distance_matrix)
            if current_best_distance < best_distance:
                best_distance = current_best_distance
            if adaptive_termination_criteria(gen, current_best_distance, best_distance):
                break
            fitness_threshold = adaptive_fitness_threshold(initial_threshold, gen, current_best_distance, best_distance)
            pop = [route for route in pop if calculate_route_length(route, distance_matrix) <= fitness_threshold]
            print("Population size after filtering:", len(pop))  
            if not pop:
                print("Population is empty after filtering")  
                return None
            best_route_index = rank_routes(pop, distance_matrix)[0][0]
            best_route = pop[best_route_index]
            #print("Best route before A* optimization:", best_route)  # Add this line
            #best_route = a_star_optimize(best_route, distance_matrix)
            #print("Best route after A* optimization:", best_route)
            final_distance = str(calculate_route_length(best_route, distance_matrix))
            print("Final distance: " + final_distance)
        else:
            print("Error: Population is empty")
            return None
        
        iteration = iteration+1
        collect_data(num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_threshold, max_k, num_partitions, final_distance, iteration)
        
    return best_route if best_route is not None else None

def decode_best_route(best_route, num_cities):
    """
    Decode the input vector x into a route.

    Parameters:
        x (list): Input vector representing the order of cities.
        num_cities (int): Number of cities in the TSP problem.

    Returns:
        list: Decoded route.
    """
    # Assuming x contains indices of cities in the order they should be visited
    route = list(best_route)
    # Ensure the route starts and ends at city 0
    if 0 not in route:
        route.insert(0, 0)
    else:
        route.remove(0)
    # Ensure the route includes all cities
    while len(route) < num_cities:
        for city in range(num_cities):
            if city not in route:
                route.append(city)
    return route

def objective_function_with(params):
    # Unpack parameters
    num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_fitness_threshold, max_k, num_partitions = params
    
    # Run the genetic algorithm with the current set of parameters
    best_route = genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_fitness_threshold, max_k, num_partitions)
    
    # Check if a valid route was found
    if best_route is not None:
        # Decode the route
        best_route = decode_best_route(best_route, num_cities)
        # Calculate the actual distance of the route
        distance_matrix = create_distance_matrix(num_cities);
        final_distance =  calculate_route_length(best_route, distance_matrix)
        # Return the actual distance for optimization
        return final_distance
    else:
        # Use a high distance value to signify an unsuccessful optimization
        return 1e9

# Define the search space for the tuning parameters
space = [
    Integer(30, 31),  # num_cities
    Integer(10, 250),   # population_size
    Integer(2, 10),     # elite_size
    Real(0.001, 0.1),   # mutation_rate
    Integer(50, 200),   # generations
    Real(0.2, 0.8),     # initial_crossover_rate
    Real(0.1, 0.5),     # final_crossover_rate
    Integer(500, 2000), # initial_fitness_threshold
    Integer(2, 5),      # range for k opt optimization
    Integer(2,4)        # num_partitions
]

# Run gp_minimize with a random forest regressor as base estimator
result = gp_minimize(
    objective_function_with,
    space,
    n_calls=100,
    base_estimator="RF"  # Using a random forest regressor
)

print("Best parameters found:")
print("num_cities:", result.x[0])
print("population_size:", result.x[1])
print("elite_size:", result.x[2])
print("mutation_rate:", result.x[3])
print("generations:", result.x[4])
print("initial_crossover_rate:", result.x[5])
print("final_crossover_rate:", result.x[6])
print("initial_fitness_threshold:", result.x[7])
print("initial_max_k", result.x[8])
print("initial_num_partitions:", result.x[9])

