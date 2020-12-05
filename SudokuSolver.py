import numpy as np

class SudokuSolver():
    def __init__(self, sudoku=np.zeros((9, 9))):
        self.sudoku = sudoku
        self.sudoku_copy = sudoku
        self.x = 0
        self.y = 0

    def set_sudoku(self, sudoku):
        self.sudoku = sudoku

    def solve(self):
        raise Exception ("solve function not implemented yet")

    def check_valid_position(self, n, x, y):
        return self.check_valid_verticals(n, x) and \
               self.check_valid_horizontal(n, y) and \
               self.check_valid_squares(n, x, y)

    def check_valid_verticals(self, n, x):
        return n not in self.sudoku[:, x]

    def check_valid_horizontal(self, n, y):
        return n not in self.sudoku[y]

    def check_valid_squares(self, n, x, y):
        x_section = x // 3
        y_section = y // 3
        for i in range (3):
            for j in range (3):
                if self.sudoku[y_section * 3 + i][x_section * 3 + j] == n:
                   return False
        return True

    def check_valid_sudoku(self):
        if self.sudoku.shape != (9, 9): return False
        for i in range(9):
            for j in range(9):
                current = self.sudoku[i, j]
                self.sudoku[i, j] = 0
                if current and not self.check_valid_position(current, j, i):
                    self.sudoku[i, j] = current
                    return False
                self.sudoku[i, j] = current
        return True

    def next(self, x, y):
        if x == 8:
            x = 0
            y += 1
        else:
            x += 1
        return x, y


    def __str__(self):
        string = ""
        string += "┌───────┬───────┬───────┐\n"
        for idx_row, row in enumerate(self.sudoku):
            if idx_row > 1 and idx_row % 3 == 0:
                string += "├───────┼───────┼───────┤\n"
            string += "│ "

            for idx_col, col in enumerate(row):
                if col:
                    string += str(col) + " "
                else:
                    string += "  "
                if idx_col > 1 and idx_col % 3 == 2:
                    string += "│ "
            string += "\n"


        string += "└───────┴───────┴───────┘"
        return string


if __name__ == "__main__":
    matrix = np.array(
        [[5, 3, 1, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],

        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],

        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]]
    )

    solver = SudokuSolver(matrix)



