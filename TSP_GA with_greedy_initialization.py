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

def create_distance_matrix(num_cities):
    """Creates a symmetric matrix of distances between cities."""
    matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
    matrix = (matrix + matrix.T) // 2  # Ensure symmetry for the TSP
    np.fill_diagonal(matrix, 0)  # Distance from a city to itself is 0
    return matrix

def calculate_route_length(route, distance_matrix):
    """Calculates the total distance of a given route."""
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1], route[0]]

def generate_initial_population(pop_size, num_cities):
    """Generates an initial population with purely random routes."""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def generate_greedy_population(pop_size, num_cities, distance_matrix):
    """Generates part of the initial population using the Nearest Neighbor algorithm."""
    population = []
    for _ in range(pop_size // 2):  # Half of the population will be generated using a greedy approach
        start_city = random.randint(0, num_cities - 1)
        route = [start_city]
        remaining_cities = set(range(num_cities)) - {start_city}
        while remaining_cities:
            last_city = route[-1]
            next_city = min(remaining_cities, key=lambda x: distance_matrix[last_city][x])
            route.append(next_city)
            remaining_cities.remove(next_city)
        population.append(route)
    # The other half of the population is generated randomly
    population.extend(generate_initial_population(pop_size // 2, num_cities))
    return population

def rank_routes(population, distance_matrix):
    """Ranks each route in the population based on its total distance."""
    fitness_results = {}
    for i, route in enumerate(population):
        fitness_results[i] = calculate_route_length(route, distance_matrix)
    return sorted(fitness_results.items(), key=lambda x: x[1])

def selection(ranked_routes, elite_size):
    """Selects the best routes to be parents for generating the next generation."""
    selection_results = [ranked_routes[i][0] for i in range(elite_size)]
    return selection_results

def mating_pool(population, selection_results):
    """Creates a mating pool from the selected routes."""
    return [population[i] for i in selection_results]

def breed(parent1, parent2):
    """Breeds two routes (parents) to produce a new route (child)."""
    child = []
    geneA, geneB = sorted(random.sample(range(len(parent1)), 2))
    childP1 = parent1[geneA:geneB]
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def breed_population(matingpool, elite_size):
    """Generates a new generation by breeding the mating pool."""
    children = matingpool[:elite_size]  # Carry forward elite routes unchanged
    pool = random.sample(matingpool, len(matingpool))
    for i in range(elite_size, len(matingpool)):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    """Mutates an individual route by swapping two cities with a given probability."""
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swapWith = int(random.random() * len(individual))
            individual[swapped], individual[swapWith] = individual[swapWith], individual[swapped]
    return individual

def mutate_population(population, mutation_rate):
    """Applies mutation across the entire population."""
    return [mutate(individual, mutation_rate) for individual in population]

def next_generation(current_gen, elite_size, mutation_rate, distance_matrix):
    """Creates the next generation from the current generation."""
    ranked_routes = rank_routes(current_gen, distance_matrix)
    selection_results = selection(ranked_routes, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def genetic_algorithm(pop_size, elite_size, mutation_rate, generations, distance_matrix):
    """Runs the genetic algorithm to find the best TSP route."""
    pop = generate_greedy_population(pop_size, len(distance_matrix), distance_matrix)
    print("Initial distance: " + str(calculate_route_length(pop[rank_routes(pop, distance_matrix)[0][0]], distance_matrix)))
    
    for i in range(generations):
        pop = next_generation(pop, elite_size, mutation_rate, distance_matrix)
    
    best_route_index = rank_routes(pop, distance_matrix)[0][0]
    best_route = pop[best_route_index]
    print("Final distance: " + str(calculate_route_length(best_route, distance_matrix)))
    return best_route

# Example usage
num_cities = 10
distance_matrix = create_distance_matrix(num_cities)
population_size = 100
elite_size = 20
mutation_rate = 0.01
generations = 500

best_route = genetic_algorithm(population_size, elite_size, mutation_rate, generations, distance_matrix)
print(f"Best Route: {best_route}")
