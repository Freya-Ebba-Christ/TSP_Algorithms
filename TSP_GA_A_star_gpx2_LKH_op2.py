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

def a_star_optimize(route, distance_matrix):
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

def create_distance_matrix(num_cities):
    matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix

def calculate_route_length(route, distance_matrix):
    route_np = np.array(route)
    return np.sum(distance_matrix[route_np[:-1], route_np[1:]]) + distance_matrix[route_np[-1], route_np[0]]

def generate_initial_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def rank_routes(population, distance_matrix):
    fitness_results = {i: calculate_route_length(population[i], distance_matrix) for i in range(len(population))}
    return sorted(fitness_results, key=fitness_results.get)

def selection(ranked_routes, elite_size):
    return ranked_routes[:elite_size]

def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]

def enhanced_gpx2_crossover(parent1, parent2, distance_matrix):
    common_paths = find_common_paths(parent1, parent2)
    offspring = []
    for path in common_paths:
        if all(city in offspring for city in path):
            continue
        offspring.extend(city for city in path if city not in offspring)
    offspring = fill_gaps_with_remaining_cities(offspring, distance_matrix)
    return offspring

def find_common_paths(parent1, parent2):
    common_edges = set(zip(parent1, parent1[1:] + parent1[:1])) & set(zip(parent2, parent2[1:] + parent2[:1]))
    visited = set()
    paths = []
    for start, _ in common_edges:
        if start in visited:
            continue
        current = start
        path = [current]
        while True:
            visited.add(current)
            next_cities = [next_city for (c, next_city) in common_edges if c == current and next_city not in path]
            if not next_cities:
                break
            current = next_cities[0]
            path.append(current)
            if path[-1] == start:
                break
        if len(path) > 1 and path[0] == path[-1]:
            paths.append(path[:-1])
    return paths

def fill_gaps_with_remaining_cities(offspring, parent1, parent2, distance_matrix):
    remaining = set(range(len(distance_matrix))) - set(offspring)
    # Ensure offspring is not empty before proceeding
    if not offspring:
        # Handle the case when offspring is empty. 
        # This could be initializing offspring with a random city or another logic that fits your algorithm.
        offspring.append(random.choice(list(remaining)))  # Example: adding a random remaining city
        remaining.remove(offspring[0])

    while remaining:
        best_pos, best_increase = 0, float('inf')  # Initialize best_pos with a default value, e.g., 0
        for city in remaining:
            for i in range(len(offspring)):
                prev_city = offspring[i - 1]
                next_city = offspring[(i + 1) % len(offspring)]
                increase = distance_matrix[prev_city][city] + distance_matrix[city][next_city] - distance_matrix[prev_city][next_city]
                if increase < best_increase:
                    best_pos, best_increase = i, increase
        # Now best_pos is guaranteed to have a value, and the following operation is safe
        offspring.insert(best_pos + 1, city)
        remaining.remove(city)
    return offspring

def mutate_individual(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def breed_population(matingpool, elite_size, distance_matrix):
    children = matingpool[:elite_size]  # Elite carryover
    for _ in range(len(matingpool) - elite_size):
        child = enhanced_gpx2_crossover(random.choice(matingpool), random.choice(matingpool), distance_matrix)
        children.append(mutate_individual(child, mutation_rate))
    return children

def next_generation(current_gen, elite_size, mutation_rate, distance_matrix):
    ranked = rank_routes(current_gen, distance_matrix)
    selection_results = selection(ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size, distance_matrix)
    return children

def lkh_inspired_optimization(route, distance_matrix):
    improved = True
    best_distance = calculate_route_length(route, distance_matrix)
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i+1, len(route) - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = calculate_route_length(new_route, distance_matrix)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    return route

def genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations):
    distance_matrix = create_distance_matrix(num_cities)
    population = generate_initial_population(pop_size, num_cities)

    for _ in range(generations):
        population = [lkh_inspired_optimization(individual, distance_matrix) for individual in population]
        population = next_generation(population, elite_size, mutation_rate, distance_matrix)

    best_route_index = rank_routes(population, distance_matrix)[0]
    best_route = population[best_route_index]

    optimized_best_route = a_star_optimize(best_route, distance_matrix)  # Implement or call A* optimization
    best_distance = calculate_route_length(optimized_best_route, distance_matrix)

    print("Optimized Best Route Distance:", best_distance)
    return optimized_best_route

# Configuration and Running the Genetic Algorithm with Enhancements
num_cities = 50
pop_size = 50
elite_size = 5
mutation_rate = 0.01
generations = 100

optimized_route = genetic_algorithm(num_cities, pop_size, elite_size, mutation_rate, generations)
print("Optimized Route:", optimized_route)
