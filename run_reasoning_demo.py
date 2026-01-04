"""
Demonstration script for various reasoning-based Gomoku agents.

This script shows how to use different reasoning strategies:
1. Chain of Thought (CoT)
2. Self-Consistency with Voting
3. Multi-Step Reasoning
4. Critic-Based Evaluation
5. Reflection

You can run different agents against each other or against the baseline LLM agent.
"""

import os
import datetime

from gomoku_env import GomokuEnv
from methods.llm_agent import LLMGomokuAgent
from methods.reasoning_agents import (
    CoTReasoningAgent,
    SelfConsistencyAgent,
    MultiStepReasoningAgent,
    CriticBasedAgent,
    ReflectionAgent,
)
from methods import api


API_URL = api.DEFAULT_API_URL
API_KEY = api.DEFAULT_API_KEY


def create_game_dir(base_dir: str = "game_records") -> str:
    """Create a directory for storing game records."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    game_dir = os.path.join(base_dir, f"game_{ts}")
    os.makedirs(game_dir, exist_ok=True)
    return game_dir


def save_board_image(env: GomokuEnv, game_dir: str, step: int) -> str:
    """Save current board state as an image."""
    img = env.render_image()
    path = os.path.join(game_dir, f"step_{step:03d}.png")
    img.save(path)
    return path


def save_final_image(env: GomokuEnv, game_dir: str) -> str:
    """Save final board state."""
    img = env.render_image()
    path = os.path.join(game_dir, "final.png")
    img.save(path)
    return path


def play_game(
    agent_a,
    agent_b,
    max_steps: int = 100,
    save_images: bool = True,
):
    """
    Play a complete game between two agents.
    
    Args:
        agent_a: Black player (goes first)
        agent_b: White player (goes second)
        max_steps: Maximum number of moves
        save_images: Whether to save board images
    """
    env = GomokuEnv()
    env.reset()
    
    if save_images:
        game_dir = create_game_dir()
        print(f"Game records will be saved to: {game_dir}")
    else:
        game_dir = None
    
    is_white_turn = False  # False -> Black(A), True -> White(B)
    step = 0
    
    if game_dir:
        save_board_image(env, game_dir, step)
    
    while step < max_steps:
        print("\n" + "=" * 60)
        print(f"Step {step + 1}")
        env.render()
        
        if env.winner is not None:
            print("Game already ended:", env.winner)
            break
        
        if is_white_turn:
            agent = agent_b
            is_white = True
        else:
            agent = agent_a
            is_white = False
        
        side_name = "White" if is_white else "Black"
        print(f"\n{agent.name} ({side_name}) is thinking...")
        
        action = agent.choose_action(env, is_white=is_white)
        print(f"\n>>> {agent.name} ({side_name}) plays at {action}")
        
        current_white, current_black, reward, done, info = env.step(action, is_white=is_white)
        step += 1
        is_white_turn = not is_white_turn
        
        if game_dir:
            save_board_image(env, game_dir, step)
        
        if done:
            print("\n" + "=" * 60)
            print("GAME OVER!")
            print("=" * 60)
            env.render()
            print(f"\nWinner: {info.get('winner')}")
            if game_dir:
                save_final_image(env, game_dir)
            break
    
    if env.winner is None:
        print("\nReached max steps without a winner. Draw.")
        if game_dir:
            save_final_image(env, game_dir)


def demo_cot_vs_baseline():
    """Chain of Thought agent vs baseline LLM agent."""
    print("\n" + "=" * 60)
    print("DEMO: Chain of Thought vs Baseline LLM")
    print("=" * 60)
    
    agent_cot = CoTReasoningAgent(
        name="CoT-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=False,
    )
    
    play_game(agent_cot, agent_baseline, max_steps=100)


def demo_self_consistency():
    """Self-Consistency agent vs baseline."""
    print("\n" + "=" * 60)
    print("DEMO: Self-Consistency (Voting) vs Baseline")
    print("=" * 60)
    
    agent_sc = SelfConsistencyAgent(
        name="SC-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        num_samples=5,  # Generate 5 candidates and vote
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=False,
    )
    
    play_game(agent_sc, agent_baseline, max_steps=100)


def demo_multistep():
    """Multi-Step reasoning agent vs baseline."""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Step Reasoning vs Baseline")
    print("=" * 60)
    
    agent_ms = MultiStepReasoningAgent(
        name="MultiStep-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=False,
    )
    
    play_game(agent_ms, agent_baseline, max_steps=100)


def demo_critic_based():
    """Critic-based agent vs baseline."""
    print("\n" + "=" * 60)
    print("DEMO: Critic-Based Reasoning vs Baseline")
    print("=" * 60)
    
    agent_critic = CriticBasedAgent(
        name="Critic-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        critic_model="gpt-4o",  # Can use a different model for critic
        num_candidates=5,
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=False,
    )
    
    play_game(agent_critic, agent_baseline, max_steps=100)


def demo_reflection():
    """Reflection agent vs baseline."""
    print("\n" + "=" * 60)
    print("DEMO: Reflection Agent vs Baseline")
    print("=" * 60)
    
    agent_reflection = ReflectionAgent(
        name="Reflection-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=False,
    )
    
    play_game(agent_reflection, agent_baseline, max_steps=100)


def demo_reasoning_battle():
    """Advanced: Two reasoning agents against each other."""
    print("\n" + "=" * 60)
    print("DEMO: CoT Agent vs Self-Consistency Agent")
    print("=" * 60)
    
    agent_cot = CoTReasoningAgent(
        name="CoT-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
    )
    
    agent_sc = SelfConsistencyAgent(
        name="SC-Agent",
        model="gpt-4o",
        api_url=API_URL,
        api_key=API_KEY,
        num_samples=3,  # Use 3 samples for faster execution
    )
    
    play_game(agent_cot, agent_sc, max_steps=100)


if __name__ == "__main__":
    import sys
    
    demos = {
        "1": ("Chain of Thought vs Baseline", demo_cot_vs_baseline),
        "2": ("Self-Consistency vs Baseline", demo_self_consistency),
        "3": ("Multi-Step Reasoning vs Baseline", demo_multistep),
        "4": ("Critic-Based vs Baseline", demo_critic_based),
        "5": ("Reflection vs Baseline", demo_reflection),
        "6": ("CoT vs Self-Consistency", demo_reasoning_battle),
    }
    
    print("\n" + "=" * 60)
    print("Reasoning-Based Gomoku Agent Demonstrations")
    print("=" * 60)
    print("\nAvailable demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print("\nUsage:")
    print("  python run_reasoning_demo.py [demo_number]")
    print("  Example: python run_reasoning_demo.py 1")
    print("\n  Or run without arguments to execute demo 1 (CoT vs Baseline)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "1"  # Default to CoT demo
    
    if choice in demos:
        name, demo_func = demos[choice]
        print(f"\nRunning: {name}\n")
        demo_func()
    else:
        print(f"\nInvalid choice: {choice}")
        print("Please choose a number from 1-6")

