"""
MCTS (Monte Carlo Tree Search) + LLM Hybrid Agent for Gomoku.

This module implements a sophisticated agent that combines:
1. Traditional MCTS algorithm for systematic game tree exploration
2. LLM-based position evaluation for strategic assessment

The hybrid approach leverages both:
- MCTS's systematic search capabilities
- LLM's pattern recognition and strategic understanding
"""

import math
import random
import time
from typing import Tuple, List, Optional, Set
from collections import defaultdict

from gomoku_env import GomokuEnv
from . import api
from .llm_agent import board_to_text


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    
    Attributes:
        state: (white_positions, black_positions, last_move, is_white_turn)
        parent: Parent node
        children: List of child nodes
        move: The move that led to this state
        visits: Number of times this node has been visited
        value: Total value accumulated (higher is better for the player)
        untried_moves: Moves that haven't been explored yet
    """
    
    def __init__(
        self,
        state: Tuple[Set, Set, Optional[Tuple[int, int]], bool],
        parent=None,
        move: Optional[Tuple[int, int]] = None,
    ):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = []
        self._initialize_untried_moves()
    
    def _initialize_untried_moves(self):
        """Initialize the list of untried moves from this state."""
        white_pos, black_pos, _, _ = self.state
        occupied = white_pos | black_pos
        
        # Generate all empty positions on the board
        all_moves = []
        for x in range(15):
            for y in range(15):
                if (x, y) not in occupied:
                    all_moves.append((x, y))
        
        # 智能排序候选走法：优先考虑靠近已有棋子、靠近中心的位置
        if occupied:
            def move_priority(move):
                """计算走法的优先级分数（越高越好）"""
                mx, my = move
                score = 0.0
                
                # 1. 靠近已有棋子的位置得分更高（距离越近分数越高）
                min_dist = float('inf')
                for ox, oy in occupied:
                    dist = abs(mx - ox) + abs(my - oy)  # 曼哈顿距离
                    min_dist = min(min_dist, dist)
                    if dist <= 2:
                        score += (3 - dist) * 10  # 距离1得20分，距离2得10分
                
                # 2. 靠近中心的位置得分更高
                center_dist = abs(mx - 7) + abs(my - 7)
                score += (14 - center_dist) * 0.5
                
                # 3. 距离最近棋子太远的位置降分
                if min_dist > 3:
                    score -= 50
                
                return score
            
            # 按优先级排序
            all_moves.sort(key=move_priority, reverse=True)
            
            # 只保留优先级最高的10-12个候选
            self.untried_moves = all_moves[:12]
        else:
            # For empty board, start near center
            center = (7, 7)
            nearby = [(x, y) for x in range(5, 10) for y in range(5, 10)]
            self.untried_moves = nearby
        
        random.shuffle(self.untried_moves)
    
    def is_fully_expanded(self) -> bool:
        """Check if all child moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        white_pos, black_pos, last_move, _ = self.state
        if last_move is None:
            return False
        
        # Check if the last move resulted in a win
        temp_env = GomokuEnv()
        temp_env.current_white = set(white_pos)
        temp_env.current_black = set(black_pos)
        
        # Determine which player made the last move
        is_white_move = last_move in white_pos
        positions = white_pos if is_white_move else black_pos
        
        return temp_env.check_winner(last_move, positions)
    
    def ucb_score(self, exploration_weight: float = 1.41) -> float:
        """
        Calculate UCB1 (Upper Confidence Bound) score.
        
        Formula: Q(v)/N(v) + c * sqrt(ln(N(parent)) / N(v))
        where Q is total value, N is visits, c is exploration weight
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self, exploration_weight: float = 1.41):
        """Select child with highest UCB score."""
        return max(self.children, key=lambda c: c.ucb_score(exploration_weight))
    
    def expand(self) -> 'MCTSNode':
        """Expand the tree by creating a new child node."""
        move = self.untried_moves.pop()
        white_pos, black_pos, _, is_white_turn = self.state
        
        # Create new state after making the move
        new_white = set(white_pos)
        new_black = set(black_pos)
        
        if is_white_turn:
            new_white.add(move)
        else:
            new_black.add(move)
        
        new_state = (new_white, new_black, move, not is_white_turn)
        child_node = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child_node)
        
        return child_node


class LLMPositionEvaluator:
    """
    Uses LLM to evaluate position quality.
    This provides strategic insight beyond random simulations.
    """
    
    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.cache = {}  # Cache evaluations to avoid redundant LLM calls
    
    def evaluate_position(
        self,
        white_pos: Set[Tuple[int, int]],
        black_pos: Set[Tuple[int, int]],
        is_white_perspective: bool,
    ) -> float:
        """
        Evaluate a position from a player's perspective.
        Returns a score between 0 and 1, where 1 is best for the player.
        """
        # Create cache key
        state_key = (
            tuple(sorted(white_pos)),
            tuple(sorted(black_pos)),
            is_white_perspective
        )
        
        if state_key in self.cache:
            return self.cache[state_key]
        
        # Create a temporary environment for board_to_text
        temp_env = GomokuEnv()
        temp_env.current_white = set(white_pos)
        temp_env.current_black = set(black_pos)
        
        board_str = board_to_text(temp_env)
        color = "White" if is_white_perspective else "Black"
        
        prompt = f"""Evaluate the following Gomoku position for {color}.
        
{board_str}

Analyze the position and provide a score from 0 to 100, where:
- 100 = {color} is winning or has overwhelming advantage
- 50 = Balanced position
- 0 = {color} is losing badly

Consider:
1. Immediate threats (4-in-a-row, 3-in-a-row)
2. Control of center and key positions
3. Potential winning patterns
4. Defensive needs

Output ONLY the numeric score (0-100), nothing else."""

        messages = [
            {"role": "system", "content": "You are a Gomoku position evaluator. Provide only numeric scores."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=10,
                timeout=30,
            )
            
            # Parse the score
            score_str = response.strip()
            score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
            normalized_score = max(0.0, min(100.0, score)) / 100.0
            
            # Cache the result
            self.cache[state_key] = normalized_score
            return normalized_score
            
        except Exception as e:
            print(f"Warning: LLM evaluation failed: {e}. Using default score 0.5")
            return 0.5


class MCTSLLMAgent:
    """
    Hybrid agent combining MCTS with LLM evaluation.
    
    The agent:
    1. Uses MCTS to explore the game tree systematically
    2. Uses LLM to evaluate leaf positions (instead of random rollouts)
    3. Selects moves based on visit counts or win rates
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
        simulations: int = 50,
        exploration_weight: float = 1.41,
        use_llm_eval: bool = True,
    ):
        """
        Initialize the MCTS + LLM agent.
        
        Args:
            name: Agent name
            model: LLM model to use for evaluation
            api_url: API endpoint
            api_key: API key
            simulations: Number of MCTS simulations per move
            exploration_weight: UCB exploration constant (higher = more exploration)
            use_llm_eval: Whether to use LLM for position evaluation (vs random)
        """
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.use_llm_eval = use_llm_eval
        
        if use_llm_eval:
            self.evaluator = LLMPositionEvaluator(model, api_url, api_key)
        else:
            self.evaluator = None
    
    def _simulate(self, node: MCTSNode, is_white_perspective: bool) -> float:
        """
        Simulate from a node to estimate its value.
        
        If use_llm_eval is True, uses LLM to evaluate the position.
        Otherwise, uses random playouts.
        """
        if self.use_llm_eval:
            # Use LLM to evaluate the position
            white_pos, black_pos, _, _ = node.state
            return self.evaluator.evaluate_position(white_pos, black_pos, is_white_perspective)
        else:
            # Traditional random simulation
            temp_env = GomokuEnv()
            white_pos, black_pos, _, is_white_turn = node.state
            temp_env.current_white = set(white_pos)
            temp_env.current_black = set(black_pos)
            
            # Quick random playout (limited to 20 moves)
            for _ in range(20):
                empty = []
                for x in range(15):
                    for y in range(15):
                        if (x, y) not in temp_env.current_white and (x, y) not in temp_env.current_black:
                            empty.append((x, y))
                
                if not empty:
                    return 0.5  # Draw
                
                move = random.choice(empty)
                positions = temp_env.current_white if is_white_turn else temp_env.current_black
                
                if is_white_turn:
                    temp_env.current_white.add(move)
                else:
                    temp_env.current_black.add(move)
                
                if temp_env.check_winner(move, positions):
                    # Return 1 if the perspective player won, 0 if opponent won
                    return 1.0 if is_white_turn == is_white_perspective else 0.0
                
                is_white_turn = not is_white_turn
            
            return 0.5  # No winner in simulation
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCB policy.
        改进：即使节点未完全扩展，也有概率选择已有子节点（基于UCB），
        这样可以深入搜索好的分支，而不是总是横向扩展。
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # 如果节点未完全扩展，有一定概率选择已有子节点
                if node.children and random.random() < 0.7:  # 70%概率选择已有子节点
                    node = node.best_child(self.exploration_weight)
                else:
                    # 30%概率扩展新节点
                    return node.expand()
            else:
                # 节点完全扩展，选择UCB最高的子节点
                node = node.best_child(self.exploration_weight)
        return node
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the simulation result up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
            # Flip value for opponent's perspective
            value = 1.0 - value
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
        max_retry: int = 3,
    ) -> Tuple[int, int]:
        """
        Choose the best move using MCTS + LLM.
        """
        print(f"\n[{self.name}] Running MCTS with {self.simulations} simulations...")
        start_time = time.time()
        
        # Create root node from current game state
        root_state = (
            set(env.current_white),
            set(env.current_black),
            env.history[-1][:2] if env.history else None,
            is_white
        )
        root = MCTSNode(root_state)
        
        # Run MCTS simulations
        for sim in range(self.simulations):
            # Selection: select a leaf node
            leaf = self._select(root)
            
            # Simulation: evaluate the leaf position
            value = self._simulate(leaf, is_white)
            
            # Backpropagation: update all nodes on the path
            self._backpropagate(leaf, value)
            
            if (sim + 1) % 10 == 0:
                print(f"  Completed {sim + 1}/{self.simulations} simulations...")
        
        # Choose the best move based on visit count (most robust)
        if not root.children:
            # Fallback: choose a random valid move
            for x in range(15):
                for y in range(15):
                    if (x, y) not in env.current_white and (x, y) not in env.current_black:
                        return (x, y)
        
        # 优先选择访问次数最多的（最被探索的）
        # 但如果访问次数相同，选择胜率最高的
        best_child = max(root.children, key=lambda c: (c.visits, c.value / max(c.visits, 1)))
        elapsed = time.time() - start_time
        
        # 打印所有子节点的统计信息（用于调试）
        if self.simulations <= 50:  # 只在模拟次数较少时打印详细信息
            print(f"  Top candidates:")
            sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)[:5]
            for i, child in enumerate(sorted_children, 1):
                win_rate = child.value / max(child.visits, 1)
                print(f"    {i}. {child.move}: visits={child.visits}, win_rate={win_rate:.3f}")
        
        print(f"[{self.name}] MCTS complete in {elapsed:.2f}s")
        print(f"  Best move: {best_child.move}")
        print(f"  Visits: {best_child.visits}, Win rate: {best_child.value/max(best_child.visits, 1):.3f}")
        
        return best_child.move

