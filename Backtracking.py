from SudokuSolver import *

class Backtracking(SudokuSolver):
    def __init__(self, sudoku=np.zeros((9, 9))):
        super().__init__(sudoku)


    def solve(self):
        if not self.check_valid_sudoku():
            return False
        self.solve_rec(0, 0)


    def solve_rec(self, x, y):
        if y > 8:
            print(self)
            return False

        while self.sudoku[y][x]:
            x, y = self.next(x, y)

            if y > 8:
                print(self)
                return False

        for n in range(1, 10):
            if self.check_valid_position(n, x, y):
                self.sudoku[y][x] = n
                self.solve_rec(*self.next(x, y))

        self.sudoku[y][x] = 0
        return


if __name__ == "__main__":

    matrix = np.array([[0, 0, 0, 7, 4, 3, 0, 9, 0],
 [0, 0, 0, 0, 0, 5, 6, 0, 0],
 [0, 3, 0, 0, 0, 0, 8, 0, 4],
 [4, 1, 0, 0, 0, 0, 0, 0, 2],
 [0, 0, 8, 0, 9, 0, 1, 0, 0],
 [2, 0, 0, 0, 0, 0, 0, 8, 3],
 [1, 0, 5, 0, 0, 0, 0, 6, 0],
 [0, 0, 2, 9, 0, 0, 0, 0, 0],
 [9, 4, 0, 5, 7, 8, 0, 0, 0]])

    solver = Backtracking(matrix)
    print(solver.solve())
