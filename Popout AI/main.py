from board import Board

def main():
    brd = Board()
    print(brd)
    print(brd.possible_moves("X"))

    brd = Board.from_string(
        """-------
        -------
        -------
        -------
        X--O-XO
        X-OXOXO""")
    print(brd)
    print(brd.possible_moves("X"))

    brd = brd.make_pop(0, "X").make_pop(0, "X")
    print(brd)
    print(brd.possible_moves("X"))

if __name__ == "__main__":
    main()