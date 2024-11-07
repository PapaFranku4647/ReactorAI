# test_adjacency.py

import unittest
from reactor_2 import *

class TestAdjacency(unittest.TestCase):
    def setUp(self):
        self.grid = initialize_grid()

    def test_center_cell(self):
        # Center cell with four neighbors
        self.assertEqual(get_adjacent_cells(1, 1, self.grid), [(0, 1), (2, 1), (1, 0), (1, 2)])

    def test_corner_cells(self):
        # Top-left corner
        self.assertEqual(get_adjacent_cells(0, 0, self.grid), [(1, 0), (0, 1)])
        # Top-right corner
        self.assertEqual(get_adjacent_cells(0, 16, self.grid), [(1, 16), (0, 15)])
        # Bottom-left corner
        self.assertEqual(get_adjacent_cells(9, 0, self.grid), [(8, 0), (9, 1)])
        # Bottom-right corner
        self.assertEqual(get_adjacent_cells(9, 16, self.grid), [(8, 16), (9, 15)])

    def test_edge_cells(self):
        # Top edge
        self.assertEqual(get_adjacent_cells(0, 5, self.grid), [(1, 5), (0, 4), (0, 6)])
        # Left edge
        self.assertEqual(get_adjacent_cells(4, 0, self.grid), [(3, 0), (5, 0), (4, 1)])
        # Right edge
        self.assertEqual(get_adjacent_cells(6, 16, self.grid), [(5, 16), (7, 16), (6, 15)])
        # Bottom edge
        self.assertEqual(get_adjacent_cells(9, 5, self.grid), [(8, 5), (9, 4), (9, 6)])

    def test_non_edge_cells(self):
        # Cells with '_' should still have correct adjacents if queried
        self.assertEqual(get_adjacent_cells(2, 1, self.grid), [(1, 1), (3, 1), (2, 0), (2, 2)])

if __name__ == '__main__':
    unittest.main()
