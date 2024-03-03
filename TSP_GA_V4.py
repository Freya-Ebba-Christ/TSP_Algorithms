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

def breed_population(matingpool, elite_size):
    children = matingpool[:elite_size]  # Carry forward elite routes unchanged
    pool = random.sample(matingpool, len(matingpool))
    for i in range(elite_size, len(matingpool)):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swapWith = int(random.random() * len(individual))
            individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]
    return individual

def mutate_population(population, mutation_rate):
    return [mutate(individual, mutation_rate) for individual in population]

def next_generation(current_gen, elite_size, mutation_rate, distance_matrix):
    ranked_routes = rank_routes(current_gen, distance_matrix)
    selection_results = selection(ranked_routes, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def calculate_route_length(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1], route[0]]


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

def create_distance_matrix(num_cities):
    matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix

def genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations):
    edges = [(i, j, np.random.randint(1, 100)) for i in range(num_cities) for j in range(i+1, num_cities)]
    pop = generate_eulerian_mst_based_population(pop_size, num_cities, edges)
    distance_matrix = create_distance_matrix(num_cities)
    
    print("Initial distance: " + str(calculate_route_length(pop[0], distance_matrix)))
    for i in range(generations):
        pop = next_generation(pop, elite_size, mutation_rate, distance_matrix)
    
    best_route_index = rank_routes(pop, distance_matrix)[0][0]
    best_route = pop[best_route_index]
    
    # Apply A* optimization to the best route
    best_route = a_star_optimize(best_route, distance_matrix)
    
    print("Final distance: " + str(calculate_route_length(best_route, distance_matrix)))
    return best_route

# Running the genetic algorithm with integrated MST and Eulerian enhancements
num_cities = 5
population_size = 20
elite_size = 5
mutation_rate = 0.01
generations = 100

best_route = genetic_algorithm(num_cities, population_size, elite_size, mutation_rate, generations)
print(f"Best Route: {best_route}")
