"""
Reasoning-enhanced prompts for Gomoku AI agents.
These prompts encourage step-by-step thinking and strategic analysis.
"""

# Chain of Thought prompts
COT_SYSTEM_PROMPT = """You are a strong Gomoku (five-in-a-row) AI expert.
You play on a 15x15 board. Coordinates are 0-based indices: (x, y).
'x' is the column index (0..14), 'y' is the row index (0..14).

Game rules:
- Players alternate placing stones on empty intersections.
- Black moves first.
- A player wins by forming 5 or more stones in a row (horizontally, vertically, or diagonally).

You should analyze the position step-by-step before making a decision.
"""

COT_USER_PROMPT_TEMPLATE = """Here is the current Gomoku board state.
Legend: 'B' = black, 'W' = white, '.' = empty.

{board_str}

It is {color}'s turn.

Please analyze this position step by step:
1. Identify any immediate threats (opponent's potential winning moves)
2. Look for your own winning opportunities
3. Consider strategic positions (forming threats, blocking opponent patterns)
4. Evaluate the best move

After your analysis, provide your final move in the format:
MOVE: x,y

Remember to output the final move clearly."""


# Multi-step reasoning prompts
MULTISTEP_SYSTEM_PROMPT = """You are an expert Gomoku AI that thinks ahead multiple moves.
You play on a 15x15 board with 0-based coordinates (x, y).

Your task is to consider potential move sequences and their outcomes."""

MULTISTEP_USER_PROMPT_TEMPLATE = """Current board state:
Legend: 'B' = black, 'W' = white, '.' = empty.

{board_str}

It is {color}'s turn.

Analyze this position by considering:
1. What are the top 3 candidate moves?
2. For each candidate, what might happen in the next 2-3 moves?
3. Which move gives you the best strategic advantage?

After your analysis, provide your chosen move in the format:
MOVE: x,y"""


# Critic prompts for evaluating moves
CRITIC_SYSTEM_PROMPT = """You are a Gomoku strategy critic and evaluator.
Your role is to analyze and score potential moves on a 15x15 board.

Evaluation criteria:
- Threat level: Does this move create immediate winning threats?
- Defense: Does this move block opponent's threats?
- Strategic value: Does this move improve overall position?
- Future potential: Does this move create multiple threats for future moves?

You should provide a numerical score from 0 to 100 for each move."""

CRITIC_USER_PROMPT_TEMPLATE = """Current board state:
{board_str}

It is {color}'s turn.

Please evaluate the following candidate move: ({x}, {y})

Provide your evaluation in this format:
1. Threat level: [score 0-25]
2. Defense value: [score 0-25]
3. Strategic value: [score 0-25]
4. Future potential: [score 0-25]
5. Overall score: [sum of above, 0-100]
6. Brief reasoning: [1-2 sentences]

SCORE: [final numeric score]"""


# Self-consistency prompt (similar to CoT but asks for direct move)
SELF_CONSISTENCY_SYSTEM_PROMPT = """You are a strong Gomoku AI player.
Board size: 15x15, coordinates: (x, y) with 0-based indexing.

Analyze the position and choose the best move."""

SELF_CONSISTENCY_USER_PROMPT_TEMPLATE = """Current board:
{board_str}

It is {color}'s turn.

Consider the current threats, opportunities, and strategic positions.
What is your chosen move?

Reply with: x,y"""


# Reflection prompt
REFLECTION_SYSTEM_PROMPT = """You are a Gomoku AI with self-reflection capabilities.
After considering a move, you can critique your own decision and potentially revise it."""

REFLECTION_USER_PROMPT_TEMPLATE = """Board state:
{board_str}

It is {color}'s turn.

First, propose your initial move and explain your reasoning.
Then, reflect on whether this move is truly optimal or if there's a better alternative.

Format:
INITIAL MOVE: x,y
REASONING: [your explanation]
REFLECTION: [critique your own move]
FINAL MOVE: x,y"""


def build_cot_prompt(board_str: str, color: str) -> str:
    """Build Chain of Thought prompt."""
    return COT_USER_PROMPT_TEMPLATE.format(board_str=board_str, color=color)


def build_multistep_prompt(board_str: str, color: str) -> str:
    """Build multi-step reasoning prompt."""
    return MULTISTEP_USER_PROMPT_TEMPLATE.format(board_str=board_str, color=color)


def build_critic_prompt(board_str: str, color: str, x: int, y: int) -> str:
    """Build critic evaluation prompt for a specific move."""
    return CRITIC_USER_PROMPT_TEMPLATE.format(board_str=board_str, color=color, x=x, y=y)


def build_self_consistency_prompt(board_str: str, color: str) -> str:
    """Build self-consistency prompt."""
    return SELF_CONSISTENCY_USER_PROMPT_TEMPLATE.format(board_str=board_str, color=color)


def build_reflection_prompt(board_str: str, color: str) -> str:
    """Build reflection prompt."""
    return REFLECTION_USER_PROMPT_TEMPLATE.format(board_str=board_str, color=color)

