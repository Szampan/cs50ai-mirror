import sys

from crossword import *

#DELETE ALL IC()
from icecream import ic     


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """

        for var in self.domains:
            to_remove = []
            for word in self.domains[var]:
                if len(word) != var.length:
                    to_remove.append(word)
            for word in to_remove:
                self.domains[var].remove(word)
            

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        overlaps = self.crossword.overlaps[x,y]
        revised = False
        for x_word in list(self.domains[x]):
            if not any(x_word[overlaps[0]] == y_word[overlaps[1]] for y_word in self.domains[y]):
                self.domains[x].remove(x_word)
                revised = True
        return revised

      
    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs:
            queue = arcs
        else:
            queue = [vars for vars in self.crossword.overlaps if self.crossword.overlaps[vars]                ]
        while queue:
            (x,y) = queue.pop(0)    # dequeue
            if not self.revise(x,y):
                return False
            
            for z in self.crossword.neighbors(x)-{y}:
                queue.append((z,x))                

        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Compare number of variables with assigned values
        if len(self.crossword.variables) == len([i for i in assignment.values() if i]):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Test distinction
        if len(assignment.values()) != len(set(assignment.values())):
            return False
        # Test length
        for i in assignment:
            if len(assignment[i]) != i.length:
                return False
        # Test conflicts
        for var_a in assignment:
            for var_b in assignment:
                if var_a != var_b:
                    overlaps = self.crossword.overlaps[var_a, var_b]
                    if overlaps:
                        if assignment[var_a][overlaps[0]] != assignment[var_b][overlaps[1]]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbors = self.crossword.neighbors(var)   # Set of variables

        def conflicts_number(value):
            counter = 0
            for neighbor in neighbors:
                for neigh_value in self.domains[neighbor]:
                    if not self.consistent({**{var:value}, **{neighbor:neigh_value}}):
                        counter += 1
            return counter

        if var:
            return sorted(self.domains[var], key=conflicts_number)
        else:
            return list()

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        remaining_values = {          # Set of dictionaries
            i: len(self.domains[i])
            for i in self.crossword.variables-assignment.keys()
        }

        # List of variables with the the minimum number of remaining values
        min_remaining_values = min(remaining_values.values())
        vars_with_smallest_domain = [k for k, v in remaining_values.items() if v==min_remaining_values]

        # Return variable with the highest degree
        def neighbors_count(var):   
            return len(self.crossword.neighbors(var))
        return max(vars_with_smallest_domain, key=neighbors_count)

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            if self.consistent({**assignment, **{var: value}}):
                assignment.update({var:value})                
                result = self.backtrack(assignment)
                if result:
                    return result
                del assignment[var]    
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
