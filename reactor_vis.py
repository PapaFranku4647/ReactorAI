import pygame
import sys

from reactor_2 import *


def visualize_grid():






    """
    Visualizes a grid using Pygame with the following features:
    - Displays cells as perfect squares.
    - Colors valid cells ('X') in green and invalid cells ('_') in gray.
    - Highlights a clicked valid cell and its adjacent cells in red.
    """
    # Initialize Pygame
    pygame.init()

    # Define the grid
    grid = initialize_grid()

    # Grid dimensions
    num_rows = len(grid)
    num_cols = len(grid[0]) if num_rows > 0 else 0

    # Define cell size and margins
    cell_size = 25  # Size of each cell in pixels
    margin = 2        # Margin between cells in pixels

    # Define colors
    COLORS = {
        "X": (0, 255, 0),          # Green for valid cells
        "_": (169, 169, 169),      # Gray for invalid cells
        "SELECTED": (255, 0, 0),   # Red for selected and adjacent cells
        "GRID_LINES": (0, 0, 0)    # Black for grid lines
    }

    # Calculate window size
    window_width = num_cols * (cell_size + margin) + margin
    window_height = num_rows * (cell_size + margin) + margin

    # Set up the display
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Interactive Grid Visualization")

    # Font for annotations
    font = pygame.font.SysFont(None, 36)

    # Variables to track selected and adjacent cells
    selected_cell = None
    adjacent_cells = []

    def draw_grid():
        """
        Draws the grid on the Pygame window.
        Highlights the selected cell and its adjacent cells if any.
        """
        for i in range(num_rows):
            for j in range(num_cols):
                # Determine the position of the cell
                x = margin + j * (cell_size + margin)
                y = margin + i * (cell_size + margin)

                # Determine the cell's color
                cell = grid[i][j]
                if selected_cell:
                    if (i, j) == selected_cell:
                        color = COLORS["SELECTED"]
                    elif (i, j) in adjacent_cells:
                        color = COLORS["SELECTED"]
                    else:
                        color = COLORS[cell]
                else:
                    color = COLORS[cell]

                # Draw the cell rectangle
                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

                # Draw grid lines (optional for better visibility)
                pygame.draw.rect(screen, COLORS["GRID_LINES"], (x, y, cell_size, cell_size), 1)

                # Render the cell's content ('X' or '_')
                if cell != "_":
                    text_surface = font.render(cell, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(x + cell_size / 2, y + cell_size / 2))
                    screen.blit(text_surface, text_rect)

        """
        Returns a list of adjacent cell positions (up, down, left, right) for a given cell (i, j).
        Only includes cells that are valid ('X').
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        adjacents = []
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < num_rows and 0 <= nj < num_cols and grid[ni][nj] == "X":
                adjacents.append((ni, nj))
        return adjacents

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Get mouse position
                mouse_x, mouse_y = event.pos

                # Determine which cell was clicked
                for i in range(num_rows):
                    for j in range(num_cols):
                        cell_x = margin + j * (cell_size + margin)
                        cell_y = margin + i * (cell_size + margin)
                        cell_rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)
                        if cell_rect.collidepoint(mouse_x, mouse_y):
                            if grid[i][j] == "X":
                                selected_cell = (i, j)
                                adjacent_cells = get_adjacent_cells(i, j, grid)
                            else:
                                # Clicked on an invalid cell; ignore or deselect
                                selected_cell = None
                                adjacent_cells = []
                            break

        # Fill the background
        screen.fill((255, 255, 255))  # White background

        # Draw the grid
        draw_grid()

        # Update the display
        pygame.display.flip()

    pygame.quit()

def visualize_individual(grid, x_positions, individual):
    """
    Visualizes a single individual by displaying the grid with tile assignments.

    Parameters:
        grid (list of lists): The original grid structure.
        x_positions (list of tuples): List of (row, col) positions marked as 'X'.
        individual (list of int): Tile assignments for each 'X' cell.

    Returns:
        None
    """
    if len(individual) != len(x_positions):
        print("Error: The number of genes does not match the number of 'X' positions.")
        print("Individual:", individual)
        print("X Positions:", x_positions)
        return
    
    # Reconstruct the grid with tile assignments
    reconstructed_grid = reconstruct_grid(individual, create_adjacency_map(x_positions, grid), grid)
    
    # Initialize Pygame
    pygame.init()

    # Grid dimensions
    num_rows = len(reconstructed_grid)
    num_cols = len(reconstructed_grid[0]) if num_rows > 0 else 0

    # Define cell size and margins
    cell_size = 50  # Increased size for better visibility
    margin = 5      # Increased margin for better spacing

    # Define colors for each tile type
    COLORS = {
        "PA": (255, 0, 0),          # Red for Power Source A
        "PB": (0, 0, 255),          # Blue for Power Source B
        "PC": (128, 0, 128),        # Purple for Power Source C
        "HSA": (255, 165, 0),       # Orange for Heat Sink A
        "HSB": (255, 215, 0),       # Gold for Heat Sink B
        "I": (0, 255, 255),         # Cyan for Iso Tile
        "_": (169, 169, 169),       # Gray for invalid cells
        "EMPTY": (211, 211, 211),   # Light Gray for EMPTY
        "GRID_LINES": (0, 0, 0)     # Black for grid lines
    }

    # Calculate window size
    window_width = num_cols * (cell_size + margin) + margin
    window_height = num_rows * (cell_size + margin) + margin

    # Set up the display
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Individual Tile Assignment Visualization")

    # Font for tile labels
    font_size = 24
    font = pygame.font.SysFont(None, font_size)

    def draw_reconstructed_grid():
        """
        Draws the reconstructed grid on the Pygame window.
        """
        for i in range(num_rows):
            for j in range(num_cols):
                # Determine the position of the cell
                x = margin + j * (cell_size + margin)
                y = margin + i * (cell_size + margin)

                # Determine the cell's color based on tile type
                cell = reconstructed_grid[i][j]
                color = COLORS.get(cell, COLORS["_"])  # Default to gray if unknown

                # Draw the cell rectangle
                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

                # Draw grid lines
                pygame.draw.rect(screen, COLORS["GRID_LINES"], (x, y, cell_size, cell_size), 2)

                # Render the cell's content (tile type abbreviation)
                if cell not in ["_", "EMPTY"]:
                    text_surface = font.render(cell, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(x + cell_size / 2, y + cell_size / 2))
                    screen.blit(text_surface, text_rect)

    # Main visualization loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background
        screen.fill((255, 255, 255))  # White background

        # Draw the grid with tile assignments
        draw_reconstructed_grid()

        # Update the display
        pygame.display.flip()

    pygame.quit()