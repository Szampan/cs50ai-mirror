"""
Tic Tac Toe Player
"""
import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if sum(x.count(EMPTY) for x in board) % 2 != 0:
        return X
    else:    
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """   
    moves = set((row, cell) for row in range(3) for cell in range(3) if board[row][cell] == EMPTY)
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """    
    new_board = copy.deepcopy(board)
    p = player(new_board)

    if new_board[action[0]][action[1]] != EMPTY:
        raise ValueError
    
    new_board[action[0]][action[1]] = p        
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    players = (X,O)

    for p in players:
        # Test horizontal
        for row in board:
            if row == [p,p,p]:
                return p

        # Test vertical (board rotated by 90*)
        for row in list(zip(*board[::-1])): 
            if row == (p,p,p):
                return p

        # Test diagonals
        if (board[0][2],board[1][1], board[2][0]) == (p,p,p) or (board[0][0],board[1][1], board[2][2]) == (p,p,p):
            return p        
    return None    


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board):
        return True  
    elif (sum(x.count(EMPTY) for x in board)) == 0:
        return True                
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    won = winner(board)
    if won == X:
        return 1
    elif won == O:
        return -1    
    return 0
    

def max_value_alpha_beta(state, alpha, beta):
    if terminal(state):
        return (utility(state), None, None)
    v = (-math.inf, None, None)
    for action in actions(state):
        v = max(v, (min_value_alpha_beta(result(state, action), alpha, beta)[:1] + action))

        if v[0] > beta:
            return v
        alpha = max(alpha, v[0])

    return v   


def min_value_alpha_beta(state, alpha, beta):  
    if terminal(state):
        return (utility(state), None, None)   
    v = (math.inf, None, None)
    for action in actions(state):
        v = min(v, (max_value_alpha_beta(result(state, action), alpha, beta)[:1] + action))

        if v[0] < alpha:
            return v      
        beta = min(beta, v[0])     

    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    if player(board) == X:  
        move = max_value_alpha_beta(board, -math.inf, math.inf)      
        return (move[1:])
    else:                       # elif???
        move = min_value_alpha_beta(board, -math.inf, math.inf)
        return (move[1:])
