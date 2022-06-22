class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        if len(board) ==0: return 0
        m = len(board)
        n = len(board[0])
        counts=0
        for row in range(0, m):
            for col in range(0, n):
                if board[row][col] == 'X' and (row == 0 or board[row-1][col] == '.') and (col == 0 or board[row][col-1] == '.'):
                    counts += 1
        return counts
                        
    def shoot(x: int, y: int):
        if (board[x][y] == "X"):
            print("It is a hit!")
        else:
            print("It is a miss. Try Again!")
            
            
