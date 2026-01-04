"""
Advanced reasoning-based Gomoku agents.
Implements various reasoning strategies including CoT, Self-Consistency, Multi-step, and Critic-based approaches.
"""

import re
from typing import Tuple, List, Optional, Dict
from collections import Counter

from gomoku_env import GomokuEnv
from . import api
from . import reasoning_prompts
from .llm_agent import board_to_text, parse_move


def parse_move_from_text(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract move from text that may contain 'MOVE: x,y' or just 'x,y'.
    More flexible than the basic parse_move.
    """
    # Try to find "MOVE: x,y" pattern first
    m = re.search(r"MOVE:\s*(-?\d+)\s*,\s*(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    
    # Fall back to basic pattern
    return parse_move(text)


def parse_score_from_text(text: str) -> Optional[float]:
    """
    Extract numeric score from critic's evaluation.
    Looks for patterns like "SCORE: 85" or "Overall score: 85"
    """
    # Try "SCORE: number" pattern
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    
    # Try "Overall score: number"
    m = re.search(r"Overall\s+score:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    
    return None


class CoTReasoningAgent:
    """
    Chain of Thought Reasoning Agent.
    Uses step-by-step thinking to analyze the board and choose moves.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
    ):
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
        max_retry: int = 3,
    ) -> Tuple[int, int]:
        """
        Use Chain of Thought reasoning to select a move.
        """
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = reasoning_prompts.build_cot_prompt(board_str, color)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.COT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for _ in range(max_retry):
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=512,  # More tokens for reasoning
            )
            
            print(f"\n[{self.name} CoT Reasoning]:")
            print(raw)
            print("-" * 40)
            
            move = parse_move_from_text(raw)
            if move and env.check_step(move):
                return move
        
        # Fallback
        return self._fallback_move(env)
    
    def _fallback_move(self, env: GomokuEnv) -> Tuple[int, int]:
        """Find first available move as fallback."""
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y
        raise RuntimeError("No legal move found.")


class SelfConsistencyAgent:
    """
    Self-Consistency Agent.
    Generates multiple reasoning paths and uses majority voting to select the best move.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
        num_samples: int = 5,
    ):
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.num_samples = num_samples
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
    ) -> Tuple[int, int]:
        """
        Generate multiple candidate moves and select by majority voting.
        """
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = reasoning_prompts.build_self_consistency_prompt(board_str, color)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.SELF_CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        candidates = []
        for i in range(self.num_samples):
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.8,  # Higher temperature for diversity
                max_tokens=256,
            )
            
            move = parse_move_from_text(raw)
            if move and env.check_step(move):
                candidates.append(move)
                print(f"[{self.name}] Sample {i+1}: {move}")
        
        if not candidates:
            print(f"[{self.name}] No valid candidates, using fallback")
            return self._fallback_move(env)
        
        # Majority voting
        counter = Counter(candidates)
        best_move = counter.most_common(1)[0][0]
        vote_count = counter[best_move]
        
        print(f"\n[{self.name} Voting Result]:")
        for move, count in counter.most_common():
            print(f"  {move}: {count} votes")
        print(f"Selected: {best_move} ({vote_count}/{len(candidates)} votes)")
        
        return best_move
    
    def _fallback_move(self, env: GomokuEnv) -> Tuple[int, int]:
        """Find first available move as fallback."""
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y
        raise RuntimeError("No legal move found.")


class MultiStepReasoningAgent:
    """
    Multi-Step Reasoning Agent.
    Considers future move sequences before making a decision.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
    ):
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
        max_retry: int = 3,
    ) -> Tuple[int, int]:
        """
        Use multi-step reasoning to analyze move sequences.
        """
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = reasoning_prompts.build_multistep_prompt(board_str, color)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.MULTISTEP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for _ in range(max_retry):
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            
            print(f"\n[{self.name} Multi-Step Reasoning]:")
            print(raw)
            print("-" * 40)
            
            move = parse_move_from_text(raw)
            if move and env.check_step(move):
                return move
        
        # Fallback
        return self._fallback_move(env)
    
    def _fallback_move(self, env: GomokuEnv) -> Tuple[int, int]:
        """Find first available move as fallback."""
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y
        raise RuntimeError("No legal move found.")


class CriticBasedAgent:
    """
    Critic-Based Reasoning Agent.
    Generates candidate moves, evaluates each with a critic model, and selects the best.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
        critic_model: Optional[str] = None,
        num_candidates: int = 5,
    ):
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.critic_model = critic_model or model  # Use same model for critic if not specified
        self.num_candidates = num_candidates
    
    def _generate_candidates(
        self,
        env: GomokuEnv,
        is_white: bool,
    ) -> List[Tuple[int, int]]:
        """Generate candidate moves using the main model."""
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = reasoning_prompts.build_self_consistency_prompt(board_str, color)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.SELF_CONSISTENCY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        candidates = []
        attempts = 0
        max_attempts = self.num_candidates * 3  # Try more times to get enough unique candidates
        
        while len(candidates) < self.num_candidates and attempts < max_attempts:
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.9,  # High temperature for diversity
                max_tokens=128,
            )
            
            move = parse_move_from_text(raw)
            if move and env.check_step(move) and move not in candidates:
                candidates.append(move)
            attempts += 1
        
        return candidates
    
    def _evaluate_move(
        self,
        env: GomokuEnv,
        is_white: bool,
        move: Tuple[int, int],
    ) -> float:
        """Use critic model to evaluate a candidate move."""
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        x, y = move
        user_prompt = reasoning_prompts.build_critic_prompt(board_str, color, x, y)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        raw = api.call_chat_completions(
            api_url=self.api_url,
            api_key=self.api_key,
            model=self.critic_model,
            messages=messages,
            temperature=0.2,  # Low temperature for consistent evaluation
            max_tokens=256,
        )
        
        score = parse_score_from_text(raw)
        if score is None:
            print(f"[{self.name}] Warning: Could not parse score for {move}, defaulting to 50")
            return 50.0
        
        print(f"[{self.name}] Evaluating {move}: score = {score}")
        return score
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
    ) -> Tuple[int, int]:
        """
        Generate candidates, evaluate each with critic, and select the best.
        """
        print(f"\n[{self.name}] Generating candidates...")
        candidates = self._generate_candidates(env, is_white)
        
        if not candidates:
            print(f"[{self.name}] No candidates generated, using fallback")
            return self._fallback_move(env)
        
        print(f"[{self.name}] Evaluating {len(candidates)} candidates...")
        evaluations: Dict[Tuple[int, int], float] = {}
        
        for move in candidates:
            score = self._evaluate_move(env, is_white, move)
            evaluations[move] = score
        
        # Select move with highest score
        best_move = max(evaluations.items(), key=lambda x: x[1])
        print(f"\n[{self.name} Critic Evaluation Results]:")
        for move, score in sorted(evaluations.items(), key=lambda x: x[1], reverse=True):
            print(f"  {move}: {score:.1f}")
        print(f"Selected: {best_move[0]} (score: {best_move[1]:.1f})")
        
        return best_move[0]
    
    def _fallback_move(self, env: GomokuEnv) -> Tuple[int, int]:
        """Find first available move as fallback."""
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y
        raise RuntimeError("No legal move found.")


class ReflectionAgent:
    """
    Reflection Agent.
    Makes an initial decision, reflects on it, and potentially revises the choice.
    """
    
    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
    ):
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
    
    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
        max_retry: int = 3,
    ) -> Tuple[int, int]:
        """
        Use reflection to potentially improve initial decision.
        """
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = reasoning_prompts.build_reflection_prompt(board_str, color)
        
        messages = [
            {"role": "system", "content": reasoning_prompts.REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for _ in range(max_retry):
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            
            print(f"\n[{self.name} Reflection]:")
            print(raw)
            print("-" * 40)
            
            # Try to extract "FINAL MOVE:" first, fall back to any move pattern
            m = re.search(r"FINAL\s+MOVE:\s*(-?\d+)\s*,\s*(-?\d+)", raw, re.IGNORECASE)
            if m:
                move = (int(m.group(1)), int(m.group(2)))
            else:
                move = parse_move_from_text(raw)
            
            if move and env.check_step(move):
                return move
        
        # Fallback
        return self._fallback_move(env)
    
    def _fallback_move(self, env: GomokuEnv) -> Tuple[int, int]:
        """Find first available move as fallback."""
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y
        raise RuntimeError("No legal move found.")

