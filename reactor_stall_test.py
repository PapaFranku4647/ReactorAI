from enum import Enum
import random
import copy
import time
import numpy as np
from scipy.stats import t
import sys

# Define Tile Types using Enum for clarity
class TileType(Enum):
    PA = 0    # Power Source A
    PB = 1    # Power Source B
    PC = 2    # Power Source C
    HSA = 3   # Heat Sink A
    HSB = 4   # Heat Sink B
    I = 5     # Iso Tile
    EMPTY = 6 # Represents an empty or non-eligible cell

HEAT_NAMES = {"PA", "PB", "PC"}    
SINK_NAMES = {"HSA", "HSB"}

# Heat generation values for power sources (positive floats)
HEAT_GENERATION = {
    TileType.PA.value: 259.5,
    TileType.PB.value: 6912.0,
    TileType.PC.value: 117188.0
}

# Heat absorption capacities for heat sinks (negative floats)
HEAT_SINK_CAPACITY = {
    TileType.HSA.value: -11556,  # Negative since they absorb heat
    TileType.HSB.value: -165302
}

IsoAmount = 0.30 # Percent that the Iso tile increases adjacent tiles' heat generation. EX: 0.05 = 5%

log_generations = False

def initialize_grid(test_grid=False):
    """
    Returns mxn array 2D grid with tile types based on the given layout.
    'X' is a valid tile, and '_' is one that cannot be placed on.
    """
    # Uncomment the original large grid for actual use
    grid = [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_", "_"],
        ["X", "_", "_", "_", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_", "_"],
        ["_", "_", "_", "_", "X", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "X", "X", "X", "X", "X", "_", "_", "X", "X", "X", "X"],
        ["_", "_", "_", "_", "_", "_", "X", "X", "X", "X", "X", "_", "_", "X", "X", "X", "X"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "X", "X", "X", "X", "X", "X", "X", "X", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "X", "X", "X", "X", "X", "X", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "X", "X", "X", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "_", "X", "_", "_", "_", "_", "_"]
    ]
    
    # Smaller grid for demonstration
    if(test_grid):
        grid = [
            ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_", "_"],
            ["X", "_", "_", "_", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_", "_"],
            ["_", "_", "_", "_", "X", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_"]
        ]
    
    return grid

def get_adjacent_cells(i, j, grid):
    """
    Returns a list of adjacent cell positions (up, down, left, right) for a given cell (i, j).

    Parameters:
        i (int): Row index of the cell.
        j (int): Column index of the cell.
        grid (list of lists): The grid structure.

    Returns:
        list of tuples: Adjacent cell positions as (row, column).
    """
    adjacent = []
    num_rows = len(grid)
    num_cols = len(grid[0]) if num_rows > 0 else 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        x, y = i + dx, j + dy
        if 0 <= x < num_rows and 0 <= y < num_cols and grid[x][y] != "_":
            adjacent.append((x, y))
    return adjacent

def get_x_positions(grid):
    """
    Identifies all positions in the grid marked with 'X'.

    Parameters:
        grid (list of lists): The grid structure.

    Returns:
        list of tuples: Positions marked with 'X' as (row, column).
    """
    x_positions = [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == "X"]
    return x_positions

def create_adjacency_map(x_positions, grid):
    """
    Creates a mapping of each 'X' cell to its adjacent cells.

    Parameters:
        x_positions (list of tuples): Positions marked with 'X'.
        grid (list of lists): The grid structure.

    Returns:
        dict: Mapping of each 'X' cell to its list of adjacent cells.
    """
    adjacency_map = {
        (i, j): get_adjacent_cells(i, j, grid) for (i, j) in x_positions
    }
    return adjacency_map

def initialize_population(pop_size, x_positions):
    """
    Initializes the population with random tile assignments.

    Parameters:
        pop_size (int): Number of individuals in the population.
        x_positions (list of tuples): List of (row, col) positions marked as 'X'.

    Returns:
        list of lists: Population where each individual is a list of tile types.
    """
    population = [[random.randint(0, 5) for _ in x_positions] for _ in range(pop_size)]
    return population

def print_individual_grid(individual, adjacency_map, grid, floats=True):
    """
    Prints the grid layout of the individual's tile assignments.
    
    Parameters:
        individual (list of int): Tile assignments for each 'X' cell.
        adjacency_map (dict): Mapping of each 'X' cell to its adjacent cells.
        grid (list of lists): The original grid structure.
        floats (bool): If True, replaces tile names with their heat generation or absorption values.
        
    Returns:
        None
    """
    # Create a new grid with assigned tile types
    display_grid = [row.copy() for row in grid]
    x_positions = list(adjacency_map.keys())
    
    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        
        if floats:
            # Show heat values instead of tile names
            if gene in HEAT_GENERATION:
                display_grid[row][col] = f"{HEAT_GENERATION[gene]:>7.1f}"
            elif gene in HEAT_SINK_CAPACITY:
                display_grid[row][col] = f"{HEAT_SINK_CAPACITY[gene]:>7.1f}"
            else:
                display_grid[row][col] = "   0.0 "  # Non-heat-producing tile
        else:
            # Show tile names
            tile_type = TileType(gene).name
            display_grid[row][col] = f"{tile_type:>7}"

    # Print the grid with either heat values or tile names
    print("\nIndividual Grid Layout:")
    for row in display_grid:
        print(" ".join(f"{cell:>7}" for cell in row))


def check_individual_fitness(individual, adjacency_map, grid):
    """
    Evaluates the fitness of an individual based on the reactor's heat balance.
    
    Parameters:
        individual (list of int): Tile assignments for each 'X' cell.
        adjacency_map (dict): Mapping of each 'X' cell to its adjacent cells.
        grid (list of lists): The reactor grid.
    
    Returns:
        float: Fitness score (higher is better).
    """
    x_positions = list(adjacency_map.keys())

    name_map = [row.copy() for row in grid]
    val_map = [row.copy() for row in grid]
    
    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        tile_type = TileType(gene)
        name_map[row][col] = tile_type.name
        
        if gene in HEAT_GENERATION:
            val_map[row][col] = HEAT_GENERATION[gene]
        elif gene in HEAT_SINK_CAPACITY:
            val_map[row][col] = HEAT_SINK_CAPACITY[gene]
        else:
            val_map[row][col] = 0.0


    # Step 4. Get Iso Multiplier Map (1 + iso_adj_map(heat_cell) * iso_multiplier)

    iso_map = [row.copy() for row in grid]
    
    for (row, col) in x_positions:
        iso_map[row][col] = 1.0
    
    for (row, col) in x_positions:
        if val_map[row][col] > 0:
            for (adj_row, adj_col) in get_adjacent_cells(row, col, grid):
                if name_map[adj_row][adj_col] == "I":
                    iso_map[row][col] += IsoAmount


    # Step 5. Apply Iso Mult to Value Map
    for (row, col) in x_positions:
        if val_map[row][col] > 0:
            val_map[row][col] *= iso_map[row][col]


    adj_sink_map = [row.copy() for row in grid]
    
    for (row, col) in x_positions:
        adj_sink_map[row][col] = 0
    
    for (row, col) in x_positions:
        if val_map[row][col] > 0:
            for (adj_row, adj_col) in get_adjacent_cells(row, col, grid):
                if val_map[adj_row][adj_col] < 0:
                    adj_sink_map[row][col] += 1


    val_map_heat_added = [row.copy() for row in val_map]
    
    for (row, col) in x_positions:
        if name_map[row][col] in HEAT_NAMES:
            for (adj_row, adj_col) in get_adjacent_cells(row, col, grid):
                if name_map[adj_row][adj_col] in SINK_NAMES:
                    val_map_heat_added[adj_row][adj_col] += val_map[row][col] / (adj_sink_map[row][col])

    generated_heat_map = [row.copy() for row in val_map]
    
    for row in range(len(generated_heat_map)):
        for idx in range(len(generated_heat_map[row])):
            generated_heat_map[row][idx] = 0

    for (row, col) in x_positions:
        if name_map[row][col] in SINK_NAMES:
            if val_map_heat_added[row][col] > 0:
                generated_heat_map[row][col] = -abs(val_map_heat_added[row][col] - val_map[row][col])
            else:
                generated_heat_map[row][col] = abs(val_map_heat_added[row][col] - val_map[row][col])

    return sum(heat for row in generated_heat_map for heat in row)


def mutate(individual, mutation_rate):
    """
    Mutates an individual by randomly changing its genes based on the mutation rate.
    
    Parameters:
        individual (list): The individual to mutate.
        mutation_rate (float): The probability of each gene being mutated.
        
    Returns:
        None: The individual is mutated in place.
    """
    num_genes = len(individual)

    for i in range(num_genes):
        if random.random() < mutation_rate:
            current_gene = individual[i]
            # Possible new tile types excluding the current one to encourage diversity
            possible_genes = [
                TileType.PA.value,
                TileType.PB.value,
                TileType.PC.value,
                TileType.HSA.value,
                TileType.HSB.value,
                TileType.I.value
            ]
            possible_genes = [gene for gene in possible_genes if gene != current_gene]
            if possible_genes:
                new_gene = random.choice(possible_genes)
                individual[i] = new_gene


def tournament_selection(population, fitness_scores, tournament_rate):
    """
    Selects an individual using tournament selection based on a tournament rate.
    
    Parameters:
        population (list): List of individuals.
        fitness_scores (dict): Dictionary mapping individual indices to their fitness scores.
        tournament_rate (float): Percentage of the population to participate in each tournament.
        
    Returns:
        list: Selected individual.
    """
    pop_size = len(population)
    # Calculate tournament size as a percentage of population size
    tournament_size = max(2, int(pop_size * tournament_rate))
    
    # Ensure tournament_size does not exceed population size
    tournament_size = min(tournament_size, pop_size)
    
    tournament = random.sample(range(pop_size), tournament_size)
    tournament_fitnesses = [(i, fitness_scores[i]) for i in tournament]
    winner_idx = max(tournament_fitnesses, key=lambda x: x[1])[0]
    return population[winner_idx]


def crossover(parent1, parent2, crossover_rate):
    """
    Performs uniform crossover between two parents.
    
    Parameters:
        parent1, parent2 (list): Parent individuals.
        crossover_rate (float): Probability of crossover occurring.
        
    Returns:
        tuple: Two offspring individuals.
    """
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2


def create_next_generation(population, fitness_scores, crossover_rate, mutation_rate, tournament_rate):
    """
    Creates the next generation using selection, crossover, and mutation.
    
    Parameters:
        population (list): Current population.
        fitness_scores (dict): Dictionary mapping individual indices to their fitness scores.
        crossover_rate (float): Crossover rate for GA.
        mutation_rate (float): Mutation rate for GA.
        tournament_rate (float): Percentage of the population participating in each tournament.
        
    Returns:
        list: New population.
    """
    new_population = []
    
    # Elitism: Keep the best individual
    best_idx = max(fitness_scores.items(), key=lambda x: x[1])[0]
    new_population.append(copy.deepcopy(population[best_idx]))
    
    # Generate rest of the population
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, fitness_scores, tournament_rate)
        parent2 = tournament_selection(population, fitness_scores, tournament_rate)
        
        # Ensure parents are different
        attempts = 0
        max_attempts = len(population)  # To prevent infinite loops
        while parent1 == parent2 and attempts < max_attempts:
            parent2 = tournament_selection(population, fitness_scores, tournament_rate)
            attempts += 1
        
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
        
        mutate(offspring1, mutation_rate)
        mutate(offspring2, mutation_rate)
        
        new_population.append(offspring1)
        if len(new_population) < len(population):
            new_population.append(offspring2)
    
    return new_population


def evaluate_population(population, adjacency_map, grid):
    """
    Evaluates the entire population and returns fitness scores.
    
    Parameters:
        population (list): List of individuals.
        adjacency_map (dict): Mapping of cell positions to adjacent cells.
        grid (list): The reactor grid.
        
    Returns:
        dict: Mapping of individual indices to fitness scores.
    """
    fitness_scores = {}
    for idx, individual in enumerate(population):
        fitness_scores[idx] = check_individual_fitness(individual, adjacency_map, grid)
    return fitness_scores

def clear_progress_bar():
    """
    Clears the progress bar from the console.
    """
    sys.stdout.write('\r')
    sys.stdout.write(' ' * 80)
    sys.stdout.write('\r')
    sys.stdout.flush()

def format_time(seconds):
    """
    Formats time in seconds into a string with days, hours, minutes, and seconds.

    Parameters:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def update_progress_bar(progress, total, elapsed_time):
    """
    Updates the progress bar with estimated time remaining, including days and hours beyond 24.

    Parameters:
        progress (int): Current progress count.
        total (int): Total count.
        elapsed_time (float): Elapsed time in seconds.
    """
    percent = (progress / total) * 100
    bar_length = 50  # Modify this to change the length of the progress bar
    filled_length = int(bar_length * progress // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    avg_time_per_unit = elapsed_time / progress if progress > 0 else 0
    remaining_time = avg_time_per_unit * (total - progress)
    time_str = format_time(remaining_time)
    
    sys.stdout.write(f'\rProgress: |{bar}| {percent:.2f}% Complete, ETA: {time_str}')
    sys.stdout.flush()

def run_genetic_algorithm(grid, population_size=100, generations=20, mutation_rate=0.05, 
                          crossover_rate=0.7, tournament_rate=0.05):
    """
    Runs the genetic algorithm to optimize reactor layout.
    
    Parameters:
        grid (list): The reactor grid.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to run.
        mutation_rate (float): Mutation rate for GA.
        crossover_rate (float): Crossover rate for GA.
        tournament_rate (float): Percentage of the population participating in each tournament.
        
    Returns:
        tuple: Best individual and its fitness score.
    """
    x_positions = get_x_positions(grid)
    adjacency_map = create_adjacency_map(x_positions, grid)
    
    # Initialize population
    population = initialize_population(population_size, x_positions)
    
    best_fitness = float('-inf')
    best_individual = None

    stall_count = 0

    # Main evolution loop
    for gen in range(generations):
        # Evaluate current population
        fitness_scores = evaluate_population(population, adjacency_map, grid)
        
        # Track best solution
        current_best_idx = max(fitness_scores.items(), key=lambda x: x[1])[0]
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = copy.deepcopy(population[current_best_idx])
            stall_count = 0
        else:
            stall_count += 1

        # if(stall_count > 0.25*(gen+1) and stall_count > 10):
        #     return best_individual, best_fitness, gen + 1
        
        # Create next generation
        population = create_next_generation(population, fitness_scores, 
                                            crossover_rate, mutation_rate, tournament_rate)

    return best_individual, best_fitness, generations


def trim_and_print(individual, adjacency_map, grid):
    """
    Removes heat sources that are not adjacent to any heat sinks and prints the result.
    
    Parameters:
        individual (list): List of tile assignments.
        adjacency_map (dict): Mapping of cell positions to adjacent cells.
        grid (list): The reactor grid.
    """
    # Create maps for analysis
    x_positions = list(adjacency_map.keys())
    modified_individual = individual.copy()
    
    # Create name map for reference
    name_map = [row.copy() for row in grid]
    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        tile_type = TileType(gene).name
        name_map[row][col] = tile_type
    
    # Check each position
    changes_made = False
    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        tile_type = TileType(gene).name
        
        # If it's a heat source, check if it's adjacent to any heat sink
        if tile_type in HEAT_NAMES:
            has_adjacent_sink = False
            for adj_row, adj_col in adjacency_map[pos]:
                adj_type = name_map[adj_row][adj_col]
                if adj_type in SINK_NAMES:
                    has_adjacent_sink = True
                    break
            
            # If no adjacent heat sink, convert to empty tile
            if not has_adjacent_sink:
                modified_individual[idx] = TileType.EMPTY.value
                changes_made = True
    
    # Print results
    print("\nOptimized Layout (Heat sources without sinks removed):")
    print_individual_grid(modified_individual, adjacency_map, grid, False)
    print("\nOptimized Heat Values:")
    print_individual_grid(modified_individual, adjacency_map, grid, True)
    
    if changes_made:
        print("\nNote: Some heat sources were removed because they had no adjacent heat sinks.")


if __name__ == "__main__":
    # Initialize the grid
    grid = initialize_grid(test_grid=False)  # Set to True for smaller grid during testing
    
    # Genetic Algorithm Parameters
    population_size = 500
    generations = 750
    mutation_rate = 0.1 # [0, 0.01, 0.02, 0.03, ..., 0.1]
    crossover_rate = 0.5   # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tournament_rate = 0.02 # Changed from tournament_size to tournament_rate (e.g., 5%)
    
    num_runs = 40
    stagnation_generations = []
    fitness_stag = []

    # Overall progress tracking
    crossover_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    mutation_rates = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    total_iterations = len(crossover_rates) * len(mutation_rates) * num_runs
    iteration_count = 0
    overall_start_time = time.time()

    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            avg_stag = 0
            avg_fitness = 0
            stagnation_generations.clear()  # Reset stagnation generations for new parameter set
            fitness_stag.clear()
            for run in range(num_runs):
                # Run the genetic algorithm and collect stagnation generation data
                best_solution, best_fitness, gen_stalled = run_genetic_algorithm(
                    grid,
                    population_size=population_size,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    tournament_rate=tournament_rate
                )
                avg_stag += gen_stalled
                avg_fitness += best_fitness

                stagnation_generations.append(gen_stalled)  # Store stagnation generation for each run
                fitness_stag.append(best_fitness)

                # Update overall progress bar
                iteration_count += 1
                elapsed_time = time.time() - overall_start_time
                update_progress_bar(iteration_count, total_iterations, elapsed_time)

            # Compute mean and standard error
            mean_stagnation = np.mean(stagnation_generations)
            mean_fitness = np.mean(fitness_stag)

            standard_error = np.std(stagnation_generations, ddof=1) / np.sqrt(num_runs)
            standard_error_fitness = np.std(fitness_stag, ddof=1) / np.sqrt(num_runs)

            # 95% confidence interval using t-distribution
            t_value = t.ppf(0.975, df=num_runs - 1)  # two-tailed for 95% CI
            confidence_interval = (mean_stagnation - t_value * standard_error, mean_stagnation + t_value * standard_error)
            confidence_interval_fitness = (mean_fitness - t_value * standard_error_fitness, mean_fitness + t_value * standard_error_fitness)

            # Average stagnation calculation
            avg_stag /= num_runs
            avg_fitness /= num_runs
            clear_progress_bar()
            print(f"Crossover Rate: {crossover_rate}, Mutation Rate: {mutation_rate}")
            print(f"Mean Stagnation: {mean_stagnation}, SE: {standard_error}, 95% CI: {confidence_interval}")
            print(f"Mean Fitness: {mean_fitness}, SE: {standard_error_fitness}, 95% CI: {confidence_interval_fitness}")
            print()

    # Clear overall progress bar after completion
    clear_progress_bar()

    # Print results
    print("\nOptimization Complete!")
    print(f"Best Fitness Score: {best_fitness}")
