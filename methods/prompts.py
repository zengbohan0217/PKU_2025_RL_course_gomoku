TEXT_SYSTEM_PROMPT = """You are a strong Gomoku (five-in-a-row) AI.
You play on a 15x15 board. Coordinates are 0-based indices: (x, y).
'x' is the column index (0..14), 'y' is the row index (0..14).
You must obey the game rules:
- Players alternate placing stones on empty intersections.
- Black moves first.
- A player wins by forming 5 or more stones in a row horizontally, vertically, or diagonally.

When given the current board position and whose turn it is,
you must choose ONE legal move and reply ONLY with:
x,y

NO other words, no explanation.
"""

VLM_SYSTEM_PROMPT = """You are a strong Gomoku (five-in-a-row) AI.
You will receive a board image (15x15 grid) showing current stones, and optional text description.
Your task: choose ONE legal move for the given side (Black or White).

Output format:
x,y

NO other words, no explanation.
"""


def build_user_prompt_from_board(board_str: str, color: str) -> str:
    """
    Build the user prompt given a board string and the side to move.
    `board_str` is typically produced by board_to_text(env),
    `color` is "Black" or "White".
    """
    return (
        "Here is the current Gomoku board state.\n"
        "Legend: 'B' = black, 'W' = white, '.' = empty.\n\n"
        f"{board_str}\n\n"
        f"It is {color}'s turn.\n"
        "Please reply with ONE legal move as 'x,y'."
    )
