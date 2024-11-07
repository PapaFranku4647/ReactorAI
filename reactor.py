import random
import numpy as np
from enum import Enum

# Define Tile Types using Enum for clarity
class TileType(Enum):
    PA = 0    # Power Source A
    PB = 1    # Power Source B
    PC = 2    # Power Source C
    HSA = 3   # Heat Sink A
    HSB = 4   # Heat Sink B
    I = 5     # Iso Tile
    EMPTY = 6 # Represents an empty or non-eligible cell

# Heat Production Constants
heat_prod_pa = 15.2
heat_prod_pb = 1160
heat_prod_pc = 75000

# Heat Sink Capacities
heat_sink_hsa = 794.1
heat_sink_hsb = 794.1

# Iso Tile Modifier
iso_tile = 0.05  # 5% increase in heat output for adjacent tiles

# Grid Initialization
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

# Positions of "X" cells
x_positions = [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == "X"]
num_cells = len(x_positions)

# Function to get adjacent cells
def get_adjacent_cells(i, j, grid):
    adjacent = []
    num_rows = len(grid)
    num_cols = len(grid[0]) if num_rows > 0 else 0

    # Up
    if i > 0:
        adjacent.append((i - 1, j))
    # Down
    if i + 1 < num_rows:
        adjacent.append((i + 1, j))
    # Left
    if j > 0:
        adjacent.append((i, j - 1))
    # Right
    if j + 1 < num_cols:
        adjacent.append((i, j + 1))

    return adjacent

# Precompute adjacent positions for each "X" cell
adjacent_positions_map = {
    (i, j): get_adjacent_cells(i, j, grid) for i, j in x_positions
}

# Genetic Algorithm parameters
population_size = 1000
generations = 100
mutation_rate = 0.01
crossover_rate = 0.7
tournament_size = 50

# Initialize population without constraints on Heat Sinks and Iso Tiles
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = []
        for _ in range(num_cells):
            # Assign a random tile type from PA, PB, PC, HSA, HSB, I
            gene = random.choice([
                TileType.PA.value,
                TileType.PB.value,
                TileType.PC.value,
                TileType.HSA.value,
                TileType.HSB.value,
                TileType.I.value
            ])
            individual.append(gene)
        population.append(individual)
    return population

# Fitness function
def fitness(individual):
    pa_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PA.value]
    pb_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PB.value]
    pc_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PC.value]
    hsa_positions = set(x_positions[k] for k, gene in enumerate(individual) if gene == TileType.HSA.value)
    hsb_positions = set(x_positions[k] for k, gene in enumerate(individual) if gene == TileType.HSB.value)
    i_positions = set(x_positions[k] for k, gene in enumerate(individual) if gene == TileType.I.value)

    # Combine all Heat Sink positions
    heat_sinks = hsa_positions.union(hsb_positions)

    # Initialize heat received by each Heat Sink
    sink_heat = {pos: 0 for pos in heat_sinks}

    # Function to get Heat Sink capacity
    def get_sink_capacity(pos):
        if pos in hsa_positions:
            return heat_sink_hsa
        elif pos in hsb_positions:
            return heat_sink_hsb
        else:
            return 0  # Non-sink positions

    # Aggregate all heat-producing sources
    sources = pa_positions + pb_positions + pc_positions

    # Distribute heat from sources to adjacent sinks
    for src_pos in sources:
        adjacent = adjacent_positions_map[src_pos]
        adjacent_sinks = [pos for pos in adjacent if pos in heat_sinks]
        n_adj_sinks = len(adjacent_sinks)

        if n_adj_sinks == 0:
            continue  # No adjacent sinks to distribute heat

        # Count adjacent Iso Tiles
        adjacent_iso = [pos for pos in adjacent_positions_map[src_pos] if pos in i_positions]
        n_adj_iso = len(adjacent_iso)

        # Determine base heat production based on tile type
        if src_pos in pa_positions:
            base_heat = heat_prod_pa
        elif src_pos in pb_positions:
            base_heat = heat_prod_pb
        elif src_pos in pc_positions:
            base_heat = heat_prod_pc
        else:
            base_heat = 0

        # Apply Iso Tile modifier
        adjusted_heat = base_heat * (1 + iso_tile * n_adj_iso)

        # Heat per sink
        heat_per_sink = adjusted_heat / n_adj_sinks

        # Distribute heat to sinks
        for sink_pos in adjacent_sinks:
            sink_heat[sink_pos] += heat_per_sink

    # Check for overloads
    overloaded = False
    for sink_pos, received_heat in sink_heat.items():
        capacity = get_sink_capacity(sink_pos)
        if received_heat > capacity:
            overloaded = True
            break

    # Calculate total harvested heat
    total_heat = sum(sink_heat.values()) if not overloaded else 0

    # Assign fitness
    if overloaded:
        fitness_value = -1e6  # Penalize overloaded configurations
    else:
        fitness_value = total_heat

    return fitness_value

# Selection - Tournament Selection
def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(population_size):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected.append(tournament[0][0])
    return selected

# Crossover - Single Point Crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_cells - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1 = parent1[:]
        child2 = parent2[:]
    return child1, child2

# Mutation - Randomly assign a new tile type without constraints
def mutate(individual):
    for i in range(num_cells):
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

# Genetic Algorithm main loop
def genetic_algorithm():
    population = initialize_population()
    best_individual = None
    best_fitness = -1e6

    for gen in range(generations):
        fitnesses = [fitness(individual) for individual in population]

        # Keep track of the best individual
        max_fit = max(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_individual = population[fitnesses.index(max_fit)]
            print(f"Generation {gen}, Best Fitness: {best_fitness}")

        # Selection
        selected = tournament_selection(population, fitnesses)

        # Crossover and Mutation
        next_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            if i + 1 < population_size:
                parent2 = selected[i + 1]
            else:
                parent2 = selected[0]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            next_population.extend([child1, child2])

        population = next_population[:population_size]

    return best_individual, best_fitness

# Function to reconstruct grid from individual
def reconstruct_grid(individual):
    # Create a deep copy of the original grid
    new_grid = [row.copy() for row in grid]
    
    # Categorize tile positions
    pa_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PA.value]
    pb_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PB.value]
    pc_positions = [x_positions[k] for k, gene in enumerate(individual) if gene == TileType.PC.value]
    hsa_positions = {x_positions[k] for k, gene in enumerate(individual) if gene == TileType.HSA.value}
    hsb_positions = {x_positions[k] for k, gene in enumerate(individual) if gene == TileType.HSB.value}
    i_positions = {x_positions[k] for k, gene in enumerate(individual) if gene == TileType.I.value}

    # Assign tile symbols
    for pos in pa_positions:
        new_grid[pos[0]][pos[1]] = 'PA'
    for pos in pb_positions:
        new_grid[pos[0]][pos[1]] = 'PB'
    for pos in pc_positions:
        new_grid[pos[0]][pos[1]] = 'PC'
    for pos in hsa_positions:
        new_grid[pos[0]][pos[1]] = 'HSA'
    for pos in hsb_positions:
        new_grid[pos[0]][pos[1]] = 'HSB'
    for pos in i_positions:
        new_grid[pos[0]][pos[1]] = ' I '

    return new_grid

# Function to print the grid with proper formatting
def print_grid(grid_to_print):
    for row in grid_to_print:
        row_str = ' '.join(f"{cell:3}" for cell in row)
        print(row_str)

# Run the Genetic Algorithm
best_individual, best_fitness = genetic_algorithm()

# Reconstruct and print the best grid
print(f"\nBest Total Harvested Heat: {best_fitness}")
best_grid = reconstruct_grid(best_individual)
print_grid(best_grid)
