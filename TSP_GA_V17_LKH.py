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
import random
import heapq

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
    selection_results = [ranked_routes[i][0] for i in range(elite_size)]
    return selection_results

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
   # Assuming 'route' is a list of city indices and 'distance_matrix' is a 2D NumPy array of distances
    # Convert route to a NumPy array for advanced indexing
    route_np = np.array(route)
    # Use advanced indexing to select the distances between consecutive cities
    # and sum them up. Also include the return to the start for a complete cycle.
    return np.sum(distance_matrix[route_np[:-1], route_np[1:]]) + distance_matrix[route_np[-1], route_np[0]]

def generate_eulerian_mst_based_population(pop_size, num_cities, edges):
    population = []
    for _ in range(pop_size):
        mst = kruskals_algorithm(edges, num_cities)
        eulerian_mst = make_graph_eulerian(mst, num_cities)
        tsp_route = eulerian_mst_to_tsp_route(eulerian_mst, num_cities)
        
        # Ensure the generated route does not contain out-of-bounds indices
        tsp_route = tsp_route[:num_cities]
        
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

def lkh_inspired_optimization(route, distance_matrix):
    """Performs an LKH-inspired optimization on the given route."""
    improved = True
    best_distance = calculate_route_length(route, distance_matrix)
    while improved:
        improved = False
        for k in range(2, len(route) - 1):  # Dynamically choose k-opt depth
            for i in range(1, len(route) - 2):
                for j in range(i+1, len(route) - 1):
                    if k == 2:  # For simplicity, only implementing 2-opt as a base case
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    else:
                        # This is where more complex LKH-like k-opt logic would go
                        # For simplicity, we're not implementing deeper k-opt exchanges here
                        continue
                    new_distance = calculate_route_length(new_route, distance_matrix)
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break  # Break to restart the search with the improved route
                if improved:
                    break
            if improved:
                break
    return route

def genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_threshold):
    edges = [(i, j, np.random.randint(1, 100)) for i in range(num_cities) for j in range(i+1, num_cities)]
    pop = generate_eulerian_mst_based_population(pop_size, num_cities, edges)
    distance_matrix = create_distance_matrix(num_cities)
    best_distance = float('inf')
    fitness_threshold = initial_threshold
    
    print("Initial distance: " + str(calculate_route_length(pop[0], distance_matrix)))
    for gen in range(generations):
        # Dynamic adjustment of crossover rate
        crossover_rate = initial_crossover_rate - ((initial_crossover_rate - final_crossover_rate) * gen / generations)
        pop = next_generation(pop, elite_size, mutation_rate, crossover_rate, distance_matrix)
        for idx in range(len(pop)):
            pop[idx] = lkh_inspired_optimization(pop[idx], distance_matrix)
        
        current_best_distance = calculate_route_length(pop[0], distance_matrix)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
        
        # Adaptive termination criteria
        if adaptive_termination_criteria(gen, current_best_distance, best_distance):
            break
        
        # Adaptive fitness threshold adjustment
        fitness_threshold = adaptive_fitness_threshold(initial_threshold, gen, current_best_distance, best_distance)
        
        # Filter population based on fitness threshold
        pop = [route for route in pop if calculate_route_length(route, distance_matrix) <= fitness_threshold]
    
    best_route_index = rank_routes(pop, distance_matrix)[0][0]
    best_route = pop[best_route_index]
    
    # Apply A* optimization to the best route
    best_route = a_star_optimize(best_route, distance_matrix)
    
    print("Final distance: " + str(calculate_route_length(best_route, distance_matrix)))
    return best_route

# Running the genetic algorithm with integrated MST and Eulerian enhancements
num_cities = 50
population_size = 50
elite_size = 5
mutation_rate = 0.01
generations = 100
initial_crossover_rate = 0.8
final_crossover_rate = 0.2
initial_fitness_threshold = 1000

best_route = genetic_algorithm(num_cities, population_size, elite_size, mutation_rate, generations, initial_crossover_rate, final_crossover_rate, initial_fitness_threshold)
print(f"Best Route: {best_route}")
