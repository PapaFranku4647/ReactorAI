from enum import Enum
import random
import copy
import time
import numpy as np

# from reactor_vis import *  # Ensure this module is available or replace with appropriate imports

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
    TileType.HSA.value: -11556.0,  # Negative since they absorb heat
    TileType.HSB.value: -165302.0
}

IsoAmount = 0.30 # Percent that the Iso tile increases adjacent tiles' heat generation. EX: 0.05 = 5%

log_generations = True

def estimate_time_remaining(start_time, current_generation, total_generations):
    elapsed_time = time.time() - start_time
    average_time_per_gen = elapsed_time / (current_generation + 1)
    remaining_time = average_time_per_gen * (total_generations - current_generation - 1)
    print(f"Estimated Time Remaining: {remaining_time:.2f} seconds")

def initialize_grid(test_grid=False):
    """
    Returns mxn NumPy 2D array grid with tile types based on the given layout.
    'X' is a valid tile, and '_' is one that cannot be placed on.
    """
    # Define the grid as a list of lists
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
    if test_grid:
        grid = [
            ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_"],
            ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "_", "_", "_"],
            ["X", "_", "_", "_", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_", "_"],
            ["_", "_", "_", "_", "X", "X", "X", "X", "X", "_", "_", "_", "X", "X", "X", "_", "_"]
        ]
    
    return np.array(grid)

def get_adjacent_cells(i, j, grid):
    """
    Returns a list of adjacent cell positions (up, down, left, right) for a given cell (i, j).

    Parameters:
        i (int): Row index of the cell.
        j (int): Column index of the cell.
        grid (np.ndarray): The grid structure.

    Returns:
        list of tuples: Adjacent cell positions as (row, column).
    """
    adjacent = []
    num_rows, num_cols = grid.shape

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        x, y = i + dx, j + dy
        if 0 <= x < num_rows and 0 <= y < num_cols and grid[x, y] != "_":
            adjacent.append((x, y))
    return adjacent

def get_x_positions(grid):
    """
    Identifies all positions in the grid marked with 'X'.

    Parameters:
        grid (np.ndarray): The grid structure.

    Returns:
        list of tuples: Positions marked with 'X' as (row, column).
    """
    x_indices = np.argwhere(grid == "X")
    x_positions = [tuple(idx) for idx in x_indices]
    return x_positions

def create_adjacency_map(x_positions, grid):
    """
    Creates a mapping of each 'X' cell to its adjacent cells.

    Parameters:
        x_positions (list of tuples): Positions marked with 'X'.
        grid (np.ndarray): The grid structure.

    Returns:
        dict: Mapping of each 'X' cell to its list of adjacent cells.
    """
    adjacency_map = {
        pos: get_adjacent_cells(pos[0], pos[1], grid) for pos in x_positions
    }
    return adjacency_map

def initialize_population(pop_size, x_positions):
    """
    Initializes the population with random tile assignments.

    Parameters:
        pop_size (int): Number of individuals in the population.
        x_positions (list of tuples): List of (row, col) positions marked as 'X'.

    Returns:
        np.ndarray: Population where each individual is a NumPy array of tile types.
    """
    # Each gene is an integer from 0 to 6 (inclusive)
    population = np.random.randint(0, 7, size=(pop_size, len(x_positions)), dtype=np.int8)
    return population

def reconstruct_grid(individual, adjacency_map, grid):
    """
    Reconstructs the grid with tile assignments based on the individual's genome.

    Parameters:
        individual (np.ndarray): Tile assignments for each 'X' cell.
        adjacency_map (dict): Mapping of each 'X' cell to its adjacent cells.
        grid (np.ndarray): The original grid structure.

    Returns:
        np.ndarray: New grid with tile assignments.
    """
    new_grid = grid.copy()

    for idx, gene in enumerate(individual):
        pos = list(adjacency_map.keys())[idx]
        row, col = pos
        tile_type = TileType(gene).name
        new_grid[row, col] = tile_type

    return new_grid

def print_individual_grid(individual, adjacency_map, grid, floats=True):
    """
    Prints the grid layout of the individual's tile assignments.
    
    Parameters:
        individual (np.ndarray): Tile assignments for each 'X' cell.
        adjacency_map (dict): Mapping of each 'X' cell to its adjacent cells.
        grid (np.ndarray): The original grid structure.
        floats (bool): If True, replaces tile names with their heat generation or absorption values.
        
    Returns:
        None
    """
    # Create a new grid with assigned tile types
    display_grid = grid.copy()
    x_positions = list(adjacency_map.keys())

    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos

        if floats:
            # Show heat values instead of tile names
            if gene in HEAT_GENERATION:
                display_grid[row, col] = f"{HEAT_GENERATION[gene]:>7.1f}"
            elif gene in HEAT_SINK_CAPACITY:
                display_grid[row, col] = f"{HEAT_SINK_CAPACITY[gene]:>7.1f}"
            else:
                display_grid[row, col] = "   0.0 "  # Non-heat-producing tile
        else:
            # Show tile names
            tile_type = TileType(gene).name
            display_grid[row, col] = f"{tile_type:>7}"

    # Print the grid with either heat values or tile names
    print("\nIndividual Grid Layout:")
    for row in display_grid:
        print(" ".join(f"{cell:>7}" for cell in row))

def check_individual_overload(individual, adjacency_map, grid):
    """
    Placeholder for the overload checking function.
    Implement the actual logic as per your reactor's requirements.
    """
    # Implement the overload checking logic here
    # This function should return True if the individual is overloaded, else False
    pass  # Replace with actual implementation

def check_individual_fitness(individual, adjacency_map, grid):
    """
    Evaluates the fitness of an individual based on the reactor's heat balance.
    
    Parameters:
        individual (np.ndarray): Tile assignments for each 'X' cell.
        adjacency_map (dict): Mapping of each 'X' cell to its adjacent cells.
        grid (np.ndarray): The reactor grid.
    
    Returns:
        float: Fitness score (higher is better).
    """
    x_positions = list(adjacency_map.keys())

    name_map = grid.copy()
    val_map = np.zeros_like(grid, dtype=np.float64)

    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        tile_type = TileType(gene)
        name_map[row, col] = tile_type.name

        if gene in HEAT_GENERATION:
            val_map[row, col] = HEAT_GENERATION[gene]
        elif gene in HEAT_SINK_CAPACITY:
            val_map[row, col] = HEAT_SINK_CAPACITY[gene]
        else:
            val_map[row, col] = 0.0

    # Step 4. Get Iso Multiplier Map (1 + iso_adj_map(heat_cell) * iso_multiplier)
    iso_map = np.ones_like(val_map, dtype=np.float64)

    for pos in x_positions:
        row, col = pos
        if val_map[row, col] > 0:
            for adj_pos in adjacency_map[pos]:
                adj_row, adj_col = adj_pos
                if name_map[adj_row, adj_col] == "I":
                    iso_map[row, col] += IsoAmount

    # Step 5. Apply Iso Mult to Value Map
    val_map *= iso_map

    adj_sink_map = np.zeros_like(val_map, dtype=np.int32)

    for pos in x_positions:
        row, col = pos
        if val_map[row, col] > 0:
            for adj_pos in adjacency_map[pos]:
                adj_row, adj_col = adj_pos
                if val_map[adj_row, adj_col] < 0:
                    adj_sink_map[row, col] += 1

    val_map_heat_added = val_map.copy()

    for pos in x_positions:
        row, col = pos
        if name_map[row, col] in HEAT_NAMES:
            sinks = adj_sink_map[row, col]
            if sinks > 0:
                for adj_pos in adjacency_map[pos]:
                    adj_row, adj_col = adj_pos
                    if name_map[adj_row, adj_col] in SINK_NAMES:
                        val_map_heat_added[adj_row, adj_col] += val_map[row, col] / sinks

    generated_heat_map = np.zeros_like(val_map_heat_added, dtype=np.float64)

    for pos in x_positions:
        row, col = pos
        if name_map[row, col] in SINK_NAMES:
            if val_map_heat_added[row, col] > 0:
                generated_heat_map[row, col] = -abs(val_map_heat_added[row, col] - val_map[row, col])
            else:
                generated_heat_map[row, col] = abs(val_map_heat_added[row, col] - val_map[row, col])

    return np.sum(generated_heat_map)

def mutate(individual, mutation_rate):
    """
    Mutates an individual by randomly changing its genes based on the mutation rate.
    
    Parameters:
        individual (np.ndarray): The individual to mutate.
        mutation_rate (float): The probability of each gene being mutated.
        
    Returns:
        None: The individual is mutated in place.
    """
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    num_mutations = np.sum(mutation_mask)
    if num_mutations > 0:
        # Possible new tile types excluding the current one to encourage diversity
        possible_genes = np.array([
            TileType.PA.value,
            TileType.PB.value,
            TileType.PC.value,
            TileType.HSA.value,
            TileType.HSB.value,
            TileType.I.value
        ], dtype=np.int8)

        for idx in np.where(mutation_mask)[0]:
            current_gene = individual[idx]
            available_genes = possible_genes[possible_genes != current_gene]
            if available_genes.size > 0:
                new_gene = np.random.choice(available_genes)
                individual[idx] = new_gene

def tournament_selection(population, fitness_scores, tournament_rate):
    """
    Selects an individual using tournament selection based on a tournament rate.
    
    Parameters:
        population (np.ndarray): List of individuals.
        fitness_scores (dict): Dictionary mapping individual indices to their fitness scores.
        tournament_rate (float): Percentage of the population to participate in each tournament.
        
    Returns:
        np.ndarray: Selected individual.
    """
    pop_size = population.shape[0]
    # Calculate tournament size as a percentage of population size
    tournament_size = max(2, int(pop_size * tournament_rate))
    
    # Ensure tournament_size does not exceed population size
    tournament_size = min(tournament_size, pop_size)
    
    tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
    tournament_fitnesses = [(i, fitness_scores[i]) for i in tournament_indices]
    winner_idx = max(tournament_fitnesses, key=lambda x: x[1])[0]
    return population[winner_idx]

def crossover(parent1, parent2, crossover_rate):
    """
    Performs uniform crossover between two parents.
    
    Parameters:
        parent1, parent2 (np.ndarray): Parent individuals.
        crossover_rate (float): Probability of crossover occurring.
        
    Returns:
        tuple: Two offspring individuals as NumPy arrays.
    """
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    mask = np.random.rand(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def create_next_generation(population, fitness_scores, crossover_rate, mutation_rate, tournament_rate):
    """
    Creates the next generation using selection, crossover, and mutation.
    
    Parameters:
        population (np.ndarray): Current population.
        fitness_scores (dict): Dictionary mapping individual indices to their fitness scores.
        crossover_rate (float): Crossover rate for GA.
        mutation_rate (float): Mutation rate for GA.
        tournament_rate (float): Percentage of the population participating in each tournament.
        
    Returns:
        np.ndarray: New population.
    """
    pop_size, num_genes = population.shape
    new_population = []

    # Elitism: Keep the best individual
    best_idx = max(fitness_scores.items(), key=lambda x: x[1])[0]
    new_population.append(population[best_idx].copy())

    # Generate rest of the population
    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, fitness_scores, tournament_rate)
        parent2 = tournament_selection(population, fitness_scores, tournament_rate)
        
        # Ensure parents are different
        attempts = 0
        max_attempts = pop_size
        while np.array_equal(parent1, parent2) and attempts < max_attempts:
            parent2 = tournament_selection(population, fitness_scores, tournament_rate)
            attempts += 1
        
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
        
        mutate(offspring1, mutation_rate)
        mutate(offspring2, mutation_rate)
        
        new_population.append(offspring1)
        if len(new_population) < pop_size:
            new_population.append(offspring2)
    
    return np.array(new_population, dtype=np.int8)

def evaluate_population(population, adjacency_map, grid):
    """
    Evaluates the entire population and returns fitness scores.
    
    Parameters:
        population (np.ndarray): List of individuals.
        adjacency_map (dict): Mapping of cell positions to adjacent cells.
        grid (np.ndarray): The reactor grid.
        
    Returns:
        dict: Mapping of individual indices to fitness scores.
    """
    fitness_scores = {}
    for idx in range(population.shape[0]):
        individual = population[idx]
        fitness_scores[idx] = check_individual_fitness(individual, adjacency_map, grid)
    return fitness_scores

def run_genetic_algorithm(grid, population_size=100, generations=20, mutation_rate=0.05, 
                          crossover_rate=0.7, tournament_rate=0.05):
    """
    Runs the genetic algorithm to optimize reactor layout.
    
    Parameters:
        grid (np.ndarray): The reactor grid.
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

    # Start time for tracking
    start_time = time.time()

    # Main evolution loop
    for gen in range(generations):
        # Evaluate current population
        fitness_scores = evaluate_population(population, adjacency_map, grid)
        
        # Track best solution
        current_best_idx = max(fitness_scores.items(), key=lambda x: x[1])[0]
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
        
        if log_generations:
            print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")
            # Print estimated time remaining
            estimate_time_remaining(start_time, gen, generations)
        
        # Create next generation
        population = create_next_generation(population, fitness_scores, 
                                            crossover_rate, mutation_rate, tournament_rate)
    
    return best_individual, best_fitness

def trim_and_print(individual, adjacency_map, grid):
    """
    Removes heat sources that are not adjacent to any heat sinks and prints the result.
    
    Parameters:
        individual (np.ndarray): List of tile assignments.
        adjacency_map (dict): Mapping of cell positions to adjacent cells.
        grid (np.ndarray): The reactor grid.
    """
    # Create maps for analysis
    x_positions = list(adjacency_map.keys())
    modified_individual = individual.copy()
    
    # Create name map for reference
    name_map = grid.copy()
    for idx, gene in enumerate(individual):
        pos = x_positions[idx]
        row, col = pos
        tile_type = TileType(gene).name
        name_map[row, col] = tile_type
    
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
                adj_type = name_map[adj_row, adj_col]
                if adj_type in SINK_NAMES:
                    has_adjacent_sink = True
                    break
            
            # If no adjacent heat sink, convert to empty tile
            if not has_adjacent_sink:
                modified_individual[idx] = TileType.EMPTY.value
                changes_made = True
    
    # Print results
    print("\nOptimized Layout (Heat sources without sinks removed):")
    print_individual_grid(modified_individual, adjacency_map, grid, floats=False)
    print("\nOptimized Heat Values:")
    print_individual_grid(modified_individual, adjacency_map, grid, floats=True)
    
    if changes_made:
        print("\nNote: Some heat sources were removed because they had no adjacent heat sinks.")

if __name__ == "__main__":
    # Initialize the grid
    grid = initialize_grid(test_grid=False)  # Set to True for smaller grid during testing
    
    # Genetic Algorithm Parameters
    population_size = 2000
    generations = 1000
    mutation_rate = 0.05
    crossover_rate = 0.9
    tournament_rate = 0.01 # Changed from tournament_size to tournament_rate (e.g., 1%)
    
    # Run the genetic algorithm
    best_solution, best_fitness = run_genetic_algorithm(
        grid,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_rate=tournament_rate
    )
    
    # Print results
    print("\nOptimization Complete!")
    print(f"Best Fitness Score: {best_fitness}")
    
    # Display the best solution
    x_positions = get_x_positions(grid)
    adjacency_map = create_adjacency_map(x_positions, grid)
    
    print("\nBest Layout:")
    print_individual_grid(best_solution, adjacency_map, grid, floats=False)
    print("\nHeat Values:")
    print_individual_grid(best_solution, adjacency_map, grid, floats=True)
    
    print("TRIMMED:")
    trim_and_print(best_solution, adjacency_map, grid)
