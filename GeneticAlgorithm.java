/*
                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.


  This version of the GNU Lesser General Public License incorporates
the terms and conditions of version 3 of the GNU General Public
License, supplemented by the additional permissions listed below.

  0. Additional Definitions.

  As used herein, "this License" refers to version 3 of the GNU Lesser
General Public License, and the "GNU GPL" refers to version 3 of the GNU
General Public License.

  "The Library" refers to a covered work governed by this License,
other than an Application or a Combined Work as defined below.

  An "Application" is any work that makes use of an interface provided
by the Library, but which is not otherwise based on the Library.
Defining a subclass of a class defined by the Library is deemed a mode
of using an interface provided by the Library.

  A "Combined Work" is a work produced by combining or linking an
Application with the Library.  The particular version of the Library
with which the Combined Work was made is also called the "Linked
Version".

  The "Minimal Corresponding Source" for a Combined Work means the
Corresponding Source for the Combined Work, excluding any source code
for portions of the Combined Work that, considered in isolation, are
based on the Application, and not on the Linked Version.

  The "Corresponding Application Code" for a Combined Work means the
object code and/or source code for the Application, including any data
and utility programs needed for reproducing the Combined Work from the
Application, but excluding the System Libraries of the Combined Work.

  1. Exception to Section 3 of the GNU GPL.

  You may convey a covered work under sections 3 and 4 of this License
without being bound by section 3 of the GNU GPL.

  2. Conveying Modified Versions.

  If you modify a copy of the Library, and, in your modifications, a
facility refers to a function or data to be supplied by an Application
that uses the facility (other than as an argument passed when the
facility is invoked), then you may convey a copy of the modified
version:

   a) under this License, provided that you make a good faith effort to
   ensure that, in the event an Application does not supply the
   function or data, the facility still operates, and performs
   whatever part of its purpose remains meaningful, or

   b) under the GNU GPL, with none of the additional permissions of
   this License applicable to that copy.

  3. Object Code Incorporating Material from Library Header Files.

  The object code form of an Application may incorporate material from
a header file that is part of the Library.  You may convey such object
code under terms of your choice, provided that, if the incorporated
material is not limited to numerical parameters, data structure
layouts and accessors, or small macros, inline functions and templates
(ten or fewer lines in length), you do both of the following:

   a) Give prominent notice with each copy of the object code that the
   Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the object code with a copy of the GNU GPL and this license
   document.

  4. Combined Works.

  You may convey a Combined Work under terms of your choice that,
taken together, effectively do not restrict modification of the
portions of the Library contained in the Combined Work and reverse
engineering for debugging such modifications, if you also do each of
the following:

   a) Give prominent notice with each copy of the Combined Work that
   the Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the Combined Work with a copy of the GNU GPL and this license
   document.

   c) For a Combined Work that displays copyright notices during
   execution, include the copyright notice for the Library among
   these notices, as well as a reference directing the user to the
   copies of the GNU GPL and this license document.

   d) Do one of the following:

       0) Convey the Minimal Corresponding Source under the terms of this
       License, and the Corresponding Application Code in a form
       suitable for, and under terms that permit, the user to
       recombine or relink the Application with a modified version of
       the Linked Version to produce a modified Combined Work, in the
       manner specified by section 6 of the GNU GPL for conveying
       Corresponding Source.

       1) Use a suitable shared library mechanism for linking with the
       Library.  A suitable mechanism is one that (a) uses at run time
       a copy of the Library already present on the user's computer
       system, and (b) will operate properly with a modified version
       of the Library that is interface-compatible with the Linked
       Version.

   e) Provide Installation Information, but only if you would otherwise
   be required to provide such information under section 6 of the
   GNU GPL, and only to the extent that such information is
   necessary to install and execute a modified version of the
   Combined Work produced by recombining or relinking the
   Application with a modified version of the Linked Version. (If
   you use option 4d0, the Installation Information must accompany
   the Minimal Corresponding Source and Corresponding Application
   Code. If you use option 4d1, you must provide the Installation
   Information in the manner specified by section 6 of the GNU GPL
   for conveying Corresponding Source.)

  5. Combined Libraries.

  You may place library facilities that are a work based on the
Library side by side in a single library together with other library
facilities that are not Applications and are not covered by this
License, and convey such a combined library under terms of your
choice, if you do both of the following:

   a) Accompany the combined library with a copy of the same work based
   on the Library, uncombined with any other library facilities,
   conveyed under the terms of this License.

   b) Give prominent notice with the combined library that part of it
   is a work based on the Library, and explaining where to find the
   accompanying uncombined form of the same work.

  6. Revised Versions of the GNU Lesser General Public License.

  The Free Software Foundation may publish revised and/or new versions
of the GNU Lesser General Public License from time to time. Such new
versions will be similar in spirit to the present version, but may
differ in detail to address new problems or concerns.

  Each version is given a distinguishing version number. If the
Library as you received it specifies that a certain numbered version
of the GNU Lesser General Public License "or any later version"
applies to it, you have the option of following the terms and
conditions either of that published version or of any later version
published by the Free Software Foundation. If the Library as you
received it does not specify a version number of the GNU Lesser
General Public License, you may choose any version of the GNU Lesser
General Public License ever published by the Free Software Foundation.

  If the Library as you received it specifies that a proxy can decide
whether future versions of the GNU Lesser General Public License shall
apply, that proxy's public statement of acceptance of any version is
permanent authorization for you to choose that version for the
Library.

 */
/**
 * Copyright (C) <2024>  <Freya Ebba Christ>
 *
 * @author Freya Ebba Christ
 */

/**
 *
 * Explanation:
 * `create_distance_matrix(num_cities)`:** Generates a random distance matrix for the cities.
 * `calculate_route_length(route, distance_matrix)`:** Calculates the total distance of a given route.
 * `generate_initial_population(pop_size, num_cities)`:** Generates the initial population of routes.
 * `rank_routes(population, distance_matrix)`:** Ranks each route in the population based on its distance (fitness).
 * `selection(ranked_routes, elite_size)`:** Selects routes for mating based on their fitness, ensuring the best routes have a higher chance of being selected.
 * `mating_pool(population, selection_results)`:** Creates a mating pool from the selected routes.
 * `breed(parent1, parent2)`:** Creates a new route (child) by combining parts of the routes from two parents.
 * `mutate(individual, mutation_rate)`:** Randomly swaps cities in a route to introduce mutations.
 * `next_generation(current_gen, elite_size, mutation_rate, distance_matrix)`:** Produces the next generation of routes from the current generation.
 * `genetic_algorithm(...)`:** Orchestrates the GA, evolving the population over a specified number of generations to find the best route.
 * This code is a basic implementation and can be further optimized or modified to include more sophisticated genetic algorithm techniques such as more complex crossover and mutation strategies, or adaptive mutation rates.
 * */
import java.util.Collections;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

public class GeneticAlgorithm {

    public static int[][] createDistanceMatrix(int numCities) {
        int[][] distanceMatrix = new int[numCities][numCities];
        Random rand = new Random();
        for (int i = 0; i < numCities; i++) {
            for (int j = i + 1; j < numCities; j++) {
                int distance = rand.nextInt(100) + 1;
                distanceMatrix[i][j] = distance;
                distanceMatrix[j][i] = distance; // Symmetric
            }
        }
        return distanceMatrix;
    }

    public static int calculateRouteLength(List<Integer> route, int[][] distanceMatrix) {
        int totalLength = 0;
        for (int i = 0; i < route.size() - 1; i++) {
            totalLength += distanceMatrix[route.get(i)][route.get(i + 1)];
        }
        totalLength += distanceMatrix[route.get(route.size() - 1)][route.get(0)]; // Add distance back to the start
        return totalLength;
    }

    public static List<List<Integer>> generateInitialPopulation(int popSize, int numCities) {
        List<List<Integer>> population = new ArrayList<>();
        for (int i = 0; i < popSize; i++) {
            List<Integer> individual = new ArrayList<>();
            for (int j = 0; j < numCities; j++) {
                individual.add(j);
            }
            Collections.shuffle(individual);
            population.add(individual);
        }
        return population;
    }

    public static List<List<Integer>> rankRoutes(List<List<Integer>> population, int[][] distanceMatrix) {
        List<List<Integer>> rankedRoutes = new ArrayList<>();
        for (List<Integer> individual : population) {
            int distance = calculateRouteLength(individual, distanceMatrix);
            List<Integer> routeWithDistance = new ArrayList<>(individual);
            routeWithDistance.add(distance);
            rankedRoutes.add(routeWithDistance);
        }
        rankedRoutes.sort((a, b) -> a.get(1) - b.get(1));
        return rankedRoutes;
    }

    public static List<Integer> selection(List<List<Integer>> rankedRoutes, int eliteSize) {
        List<Integer> selectionResults = new ArrayList<>();
        for (int i = 0; i < eliteSize; i++) {
            selectionResults.add(rankedRoutes.get(i).get(0));
        }
        Random rand = new Random();
        for (int i = 0; i < rankedRoutes.size() - eliteSize; i++) {
            double pick = rand.nextDouble();
            double cumulativeProbability = 0;
            for (List<Integer> route : rankedRoutes) {
                cumulativeProbability += (double) route.get(1) / rankedRoutes.stream().mapToInt(r -> r.get(1)).sum();
                if (pick <= cumulativeProbability) {
                    selectionResults.add(route.get(0));
                    break;
                }
            }
        }
        return selectionResults;
    }

    public static List<List<Integer>> matingPool(List<List<Integer>> population, List<Integer> selectionResults) {
        List<List<Integer>> pool = new ArrayList<>();
        for (int i : selectionResults) {
            pool.add(population.get(i));
        }
        return pool;
    }

    public static List<Integer> breed(List<Integer> parent1, List<Integer> parent2) {
        int startPos = (int) (Math.random() * parent1.size());
        int endPos = (int) (Math.random() * parent1.size());

        List<Integer> child = new ArrayList<>(Collections.nCopies(parent1.size(), -1));
        for (int i = startPos; i < endPos; i++) {
            child.set(i, parent1.get(i));
        }

        for (int i = 0; i < parent2.size(); i++) {
            int city = parent2.get(i);
            if (!child.contains(city)) {
                for (int j = 0; j < child.size(); j++) {
                    if (child.get(j) == -1) {
                        child.set(j, city);
                        break;
                    }
                }
            }
        }
        return child;
    }

    public static List<List<Integer>> breedPopulation(List<List<Integer>> matingpool, int eliteSize) {
        List<List<Integer>> children = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < eliteSize; i++) {
            children.add(matingpool.get(i));
        }
        for (int i = eliteSize; i < matingpool.size(); i++) {
            int randomIndex1 = rand.nextInt(matingpool.size());
            int randomIndex2 = rand.nextInt(matingpool.size());
            List<Integer> child = breed(matingpool.get(randomIndex1), matingpool.get(randomIndex2));
            children.add(child);
        }
        return children;
    }

    public static List<Integer> mutate(List<Integer> individual, double mutationRate) {
        Random rand = new Random();
        for (int i = 0; i < individual.size(); i++) {
            if (rand.nextDouble() < mutationRate) {
                int swapWith = rand.nextInt(individual.size());
                int city1 = individual.get(i);
                int city2 = individual.get(swapWith);
                individual.set(i, city2);
                individual.set(swapWith, city1);
            }
        }
        return individual;
    }

    public static List<List<Integer>> mutatePopulation(List<List<Integer>> population, double mutationRate) {
        List<List<Integer>> mutatedPopulation = new ArrayList<>();
        for (List<Integer> individual : population) {
            mutatedPopulation.add(mutate(individual, mutationRate));
        }
        return mutatedPopulation;
    }

    public static List<List<Integer>> nextGeneration(List<List<Integer>> currentGen, int eliteSize, double mutationRate, int[][] distanceMatrix) {
        List<List<Integer>> rankedRoutes = rankRoutes(currentGen, distanceMatrix);
        List<Integer> selectionResults = selection(rankedRoutes, eliteSize);
        List<List<Integer>> matingpool = matingPool(currentGen, selectionResults);
        List<List<Integer>> children = breedPopulation(matingpool, eliteSize);
        return mutatePopulation(children, mutationRate);
    }

    public static List<Integer> geneticAlgorithm(int popSize, int eliteSize, double mutationRate, int generations, int[][] distanceMatrix) {
        List<List<Integer>> pop = generateInitialPopulation(popSize, distanceMatrix.length);
        System.out.println("Initial distance: " + rankRoutes(pop, distanceMatrix).get(0).get(1));
        for (int i = 0; i < generations; i++) {
            pop = nextGeneration(pop, eliteSize, mutationRate, distanceMatrix);
        }
        int bestRouteIndex = rankRoutes(pop, distanceMatrix).get(0).get(0);
        return pop.get(bestRouteIndex);
    }

    public static void main(String[] args) {
        int numCities = 10;
        int[][] distanceMatrix = createDistanceMatrix(numCities);
        int populationSize = 100;
        int eliteSize = 20;
        double mutationRate = 0.01;
        int generations = 500;

        List<Integer> bestRoute = geneticAlgorithm(populationSize, eliteSize, mutationRate, generations, distanceMatrix);
        System.out.println("Final distance: " + calculateRouteLength(bestRoute, distanceMatrix));
        System.out.println("Best route: " + bestRoute);
    }
}
