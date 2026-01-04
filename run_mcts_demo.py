"""
Demonstration script for MCTS + LLM Hybrid Agent.

This script shows how to use the MCTSLLMAgent in various scenarios:
1. MCTS+LLM vs Baseline LLM
2. MCTS+LLM vs MCTS (without LLM evaluation)
3. MCTS+LLM vs MCTS+LLM (both using hybrid approach)

The MCTS agent combines systematic tree search with LLM strategic evaluation,
providing a more robust playing strategy than either approach alone.
"""

import os
import datetime
import argparse

from gomoku_env import GomokuEnv
from methods.llm_agent import LLMGomokuAgent
from methods.mcts_agent import MCTSLLMAgent
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
    
    Returns:
        winner: "Black", "White", or "Draw"
    """
    env = GomokuEnv()
    env.reset()
    
    if save_images:
        game_dir = create_game_dir()
        print(f"\n{'='*60}")
        print(f"Game records will be saved to: {game_dir}")
        print(f"{'='*60}\n")
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
            side_name = "White"
        else:
            agent = agent_a
            is_white = False
            side_name = "Black"
        
        print(f"\n{agent.name} ({side_name}) is thinking...")
        
        try:
            action = agent.choose_action(env, is_white=is_white)
            print(f"{agent.name} ({side_name}) plays at {action}")
        except Exception as e:
            print(f"Error: {agent.name} failed to choose action: {e}")
            print("Using fallback: first available position")
            # Fallback to first available position
            for x in range(15):
                for y in range(15):
                    if (x, y) not in env.current_white and (x, y) not in env.current_black:
                        action = (x, y)
                        break
        
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
            winner = info.get("winner", "Unknown")
            print(f"\nWinner: {winner}")
            
            if game_dir:
                save_final_image(env, game_dir)
            
            return winner
    
    if env.winner is None:
        print("\n" + "=" * 60)
        print("GAME OVER - Maximum steps reached")
        print("=" * 60)
        env.render()
        
        if game_dir:
            save_final_image(env, game_dir)
        
        return "Draw"
    
    return env.winner


def demo_mcts_vs_baseline(model: str = "gpt-4o", simulations: int = 50):
    """
    Demo: MCTS+LLM agent vs baseline LLM agent.
    This demonstrates the advantage of systematic search.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: MCTS+LLM Agent vs Baseline LLM Agent")
    print("=" * 60)
    print(f"MCTS agent uses {simulations} simulations per move")
    print(f"Baseline agent uses simple prompt-based decision")
    print()
    
    agent_mcts = MCTSLLMAgent(
        name="MCTS+LLM",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
        simulations=simulations,
        use_llm_eval=True,
    )
    
    agent_baseline = LLMGomokuAgent(
        name="Baseline-LLM",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
    )
    
    winner = play_game(agent_mcts, agent_baseline, max_steps=100)
    
    print("\n" + "=" * 60)
    print(f"Result: {winner} wins!")
    print("=" * 60)
    
    return winner


def demo_mcts_with_vs_without_llm(model: str = "gpt-4o", simulations: int = 50):
    """
    Demo: MCTS+LLM vs MCTS (without LLM evaluation).
    This demonstrates the value of LLM strategic evaluation.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: MCTS+LLM vs MCTS (random rollouts)")
    print("=" * 60)
    print(f"Both agents use {simulations} simulations")
    print(f"MCTS+LLM uses LLM for position evaluation")
    print(f"MCTS uses random playouts")
    print()
    
    agent_mcts_llm = MCTSLLMAgent(
        name="MCTS+LLM",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
        simulations=simulations,
        use_llm_eval=True,
    )
    
    agent_mcts_random = MCTSLLMAgent(
        name="MCTS-Random",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
        simulations=simulations,
        use_llm_eval=False,
    )
    
    winner = play_game(agent_mcts_llm, agent_mcts_random, max_steps=100)
    
    print("\n" + "=" * 60)
    print(f"Result: {winner} wins!")
    print("=" * 60)
    
    return winner


def demo_mcts_vs_mcts(model: str = "gpt-4o", simulations: int = 50):
    """
    Demo: MCTS+LLM vs MCTS+LLM.
    Both agents use the hybrid approach.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: MCTS+LLM vs MCTS+LLM")
    print("=" * 60)
    print(f"Both agents use {simulations} simulations + LLM evaluation")
    print()
    
    agent_a = MCTSLLMAgent(
        name="MCTS+LLM-A",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
        simulations=simulations,
        use_llm_eval=True,
    )
    
    agent_b = MCTSLLMAgent(
        name="MCTS+LLM-B",
        model=model,
        api_url=API_URL,
        api_key=API_KEY,
        simulations=simulations,
        use_llm_eval=True,
    )
    
    winner = play_game(agent_a, agent_b, max_steps=100)
    
    print("\n" + "=" * 60)
    print(f"Result: {winner} wins!")
    print("=" * 60)
    
    return winner


def run_benchmark(model: str = "gpt-4o", simulations: int = 30, num_games: int = 3):
    """
    Run a small benchmark comparing MCTS+LLM against baseline.
    
    Args:
        model: LLM model to use
        simulations: Number of MCTS simulations per move
        num_games: Number of games to play
    """
    print("\n" + "=" * 60)
    print("BENCHMARK: MCTS+LLM vs Baseline LLM")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Simulations: {simulations}")
    print(f"Number of games: {num_games}")
    print()
    
    results = {"Black": 0, "White": 0, "Draw": 0}
    
    for game_num in range(num_games):
        print(f"\n{'#'*60}")
        print(f"GAME {game_num + 1}/{num_games}")
        print(f"{'#'*60}")
        
        # Alternate who plays Black/White
        if game_num % 2 == 0:
            print("MCTS+LLM plays Black, Baseline plays White")
            agent_black = MCTSLLMAgent(
                name="MCTS+LLM",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=True,
            )
            agent_white = LLMGomokuAgent(
                name="Baseline",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
            )
            mcts_is_black = True
        else:
            print("Baseline plays Black, MCTS+LLM plays White")
            agent_black = LLMGomokuAgent(
                name="Baseline",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
            )
            agent_white = MCTSLLMAgent(
                name="MCTS+LLM",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=True,
            )
            mcts_is_black = False
        
        winner = play_game(agent_black, agent_white, max_steps=100, save_images=False)
        results[winner] += 1
        
        # Attribute win to correct agent
        if winner == "Black":
            mcts_wins = mcts_is_black
        elif winner == "White":
            mcts_wins = not mcts_is_black
        else:
            mcts_wins = None
        
        if mcts_wins is True:
            print(f"\n>>> Game {game_num + 1} Result: MCTS+LLM WINS <<<")
        elif mcts_wins is False:
            print(f"\n>>> Game {game_num + 1} Result: Baseline WINS <<<")
        else:
            print(f"\n>>> Game {game_num + 1} Result: DRAW <<<")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Black wins: {results['Black']}")
    print(f"White wins: {results['White']}")
    print(f"Draws: {results['Draw']}")
    print()
    
    # Count MCTS wins more carefully
    mcts_wins = 0
    baseline_wins = 0
    for game_num in range(num_games):
        winner = ["Black", "White", "Draw"][game_num % 3]  # This is simplified
        # In reality, we'd track this properly during the games
    
    print(f"Total games: {num_games}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS+LLM Gomoku Agent Demo")
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Which demo to run (1: vs baseline, 2: with/without LLM, 3: vs self)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=30,
        help="Number of MCTS simulations per move"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode (multiple games)"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=3,
        help="Number of games for benchmark"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(
            model=args.model,
            simulations=args.simulations,
            num_games=args.num_games
        )
    else:
        if args.demo == 1:
            demo_mcts_vs_baseline(model=args.model, simulations=args.simulations)
        elif args.demo == 2:
            demo_mcts_with_vs_without_llm(model=args.model, simulations=args.simulations)
        elif args.demo == 3:
            demo_mcts_vs_mcts(model=args.model, simulations=args.simulations)

