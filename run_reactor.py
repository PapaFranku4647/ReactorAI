import numpy as np
import random
import pandas as pd
import time
from datetime import timedelta
from reactor_2 import initialize_grid, get_x_positions, create_adjacency_map, run_genetic_algorithm

# Variable for the filename
filename = 'hyperparameter_results_0.csv'

# Function to generate random hyperparameter combinations
def generate_hyperparameters(num_samples=100):
    hyperparameters = []
    for _ in range(num_samples):
        population = random.randint(100, 1000)
        generations = random.randint(10, 100)
        mutation_rate = round(random.uniform(0, 0.25), 3)
        crossover_rate = round(random.uniform(0.5, 1.0), 3)
        tournament_rate = round(random.uniform(0.01, 0.75), 3)
        
        hyperparameters.append((population, generations, mutation_rate, crossover_rate, tournament_rate))
    return hyperparameters

def get_fitness_after_training(pop_size=100, gens=20, mut_rate=0.05, cross_rate=0.7, tour_rate=0.05):
    """
    Runs the genetic algorithm with the given hyperparameters and returns the best fitness score.

    Parameters:
        pop_size (int): Population size for the genetic algorithm.
        gens (int): Number of generations to run.
        mut_rate (float): Mutation rate.
        cross_rate (float): Crossover rate.
        tour_rate (float): Tournament selection rate.

    Returns:
        float: The best fitness score found after training.
    """
    # Initialize the grid
    grid = initialize_grid(test_grid=False)

    # Run the genetic algorithm
    _, best_fitness = run_genetic_algorithm(
        grid,
        population_size=pop_size,
        generations=gens,
        mutation_rate=mut_rate,
        crossover_rate=cross_rate,
        tournament_rate=tour_rate
    )

    return best_fitness

def evaluate_hyperparameters(hyperparameter_sets, runs_per_set=3):
    results = []
    total_runs = len(hyperparameter_sets) * runs_per_set
    start_time = time.time()

    for idx, hyperparams in enumerate(hyperparameter_sets):
        population, generations, mutation_rate, crossover_rate, tournament_rate = hyperparams
        fitness_scores = []

        for _ in range(runs_per_set):
            fitness = get_fitness_after_training(
                pop_size=population,
                gens=generations,
                mut_rate=mutation_rate,
                cross_rate=crossover_rate,
                tour_rate=tournament_rate
            )
            fitness_scores.append(fitness)
        
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        results.append({
            'population': population,
            'generations': generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'tournament_rate': tournament_rate,
            'average_fitness': average_fitness
        })
        
        elapsed_time = time.time() - start_time
        runs_completed = (idx + 1) * runs_per_set
        time_per_run = elapsed_time / runs_completed
        remaining_runs = total_runs - runs_completed
        estimated_time_remaining = timedelta(seconds=int(time_per_run * remaining_runs))
        
        print(f"Hyperparameters: {hyperparams} - Average Fitness: {average_fitness}")
        print(f"Estimated time remaining: {estimated_time_remaining}")

    return results

if __name__ == "__main__":
    hyperparameter_sets = generate_hyperparameters(num_samples=20)  # Adjust num_samples as needed
    data = evaluate_hyperparameters(hyperparameter_sets, runs_per_set=5)  # Run each set 5 times for averaging

    # Convert results to a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV for future analysis
    df.to_csv(filename, index=False)

    print(f"Data collection complete. Results saved to '{filename}'.")
