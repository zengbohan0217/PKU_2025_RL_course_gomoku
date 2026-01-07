"""
实验数据收集脚本 - 用于PPT展示

运行多个对比实验，收集胜率、平均步数、平均用时等数据。
"""

import time
import json
from datetime import datetime
from collections import defaultdict

from gomoku_env import GomokuEnv
from methods.llm_agent import LLMGomokuAgent
from methods.mcts_agent import MCTSLLMAgent
from methods import api


# 使用提供的 API 配置
API_URL = "http://123.129.219.111:3000/v1"
API_KEY = "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D"


def play_game_with_stats(
    agent_a,
    agent_b,
    max_steps: int = 100,
    save_images: bool = False,
):
    """
    对弈并收集统计数据
    
    Returns:
        dict: {
            'winner': str,
            'total_steps': int,
            'agent_a_time': float,
            'agent_b_time': float,
            'agent_a_moves': int,
            'agent_b_moves': int,
        }
    """
    env = GomokuEnv()
    env.reset()
    
    is_white_turn = False
    step = 0
    
    stats = {
        'agent_a_time': 0.0,
        'agent_b_time': 0.0,
        'agent_a_moves': 0,
        'agent_b_moves': 0,
    }
    
    while step < max_steps:
        if env.winner is not None:
            break
        
        if is_white_turn:
            agent = agent_b
            is_white = True
            is_agent_a = False
        else:
            agent = agent_a
            is_white = False
            is_agent_a = True
        
        # 计时
        start_time = time.time()
        try:
            action = agent.choose_action(env, is_white=is_white)
        except Exception as e:
            print(f"Error: {e}")
            # Fallback
            for x in range(15):
                for y in range(15):
                    if (x, y) not in env.current_white and (x, y) not in env.current_black:
                        action = (x, y)
                        break
        
        elapsed = time.time() - start_time
        
        if is_agent_a:
            stats['agent_a_time'] += elapsed
            stats['agent_a_moves'] += 1
        else:
            stats['agent_b_time'] += elapsed
            stats['agent_b_moves'] += 1
        
        env.step(action, is_white=is_white)
        step += 1
        is_white_turn = not is_white_turn
        
        if env.winner is not None:
            break
    
    winner = env.winner if env.winner else "Draw"
    stats['winner'] = winner
    stats['total_steps'] = step
    
    return stats


def experiment_mcts_vs_baseline(
    model: str = "gpt-4o",
    simulations: int = 30,
    num_games: int = 5,
):
    """
    实验1: MCTS+LLM vs Baseline LLM
    """
    print("\n" + "=" * 70)
    print("实验1: MCTS+LLM vs Baseline LLM")
    print("=" * 70)
    print(f"模型: {model}")
    print(f"MCTS模拟次数: {simulations}")
    print(f"对局数: {num_games}")
    print()
    
    results = {
        'mcts_wins': 0,
        'baseline_wins': 0,
        'draws': 0,
        'mcts_avg_time': [],
        'baseline_avg_time': [],
        'avg_steps': [],
    }
    
    for game_num in range(num_games):
        print(f"\n{'#'*70}")
        print(f"对局 {game_num + 1}/{num_games}")
        print(f"{'#'*70}")
        
        # 交替执黑/白
        if game_num % 2 == 0:
            print("MCTS+LLM 执黑，Baseline 执白")
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
            print("Baseline 执黑，MCTS+LLM 执白")
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
        
        stats = play_game_with_stats(agent_black, agent_white, max_steps=100, save_images=False)
        
        # 判断胜负
        if stats['winner'] == "Black":
            winner_is_mcts = mcts_is_black
        elif stats['winner'] == "White":
            winner_is_mcts = not mcts_is_black
        else:
            winner_is_mcts = None
        
        if winner_is_mcts:
            results['mcts_wins'] += 1
            print(f"✓ MCTS+LLM 获胜")
        elif winner_is_mcts is False:
            results['baseline_wins'] += 1
            print(f"✗ Baseline 获胜")
        else:
            results['draws'] += 1
            print(f"○ 平局")
        
        # 收集时间数据
        if mcts_is_black:
            mcts_time = stats['agent_a_time'] / stats['agent_a_moves'] if stats['agent_a_moves'] > 0 else 0
            baseline_time = stats['agent_b_time'] / stats['agent_b_moves'] if stats['agent_b_moves'] > 0 else 0
        else:
            mcts_time = stats['agent_b_time'] / stats['agent_b_moves'] if stats['agent_b_moves'] > 0 else 0
            baseline_time = stats['agent_a_time'] / stats['agent_a_moves'] if stats['agent_a_moves'] > 0 else 0
        
        results['mcts_avg_time'].append(mcts_time)
        results['baseline_avg_time'].append(baseline_time)
        results['avg_steps'].append(stats['total_steps'])
        
        print(f"  步数: {stats['total_steps']}")
        print(f"  MCTS平均用时/步: {mcts_time:.2f}秒")
        print(f"  Baseline平均用时/步: {baseline_time:.2f}秒")
    
    # 计算平均值
    results['mcts_avg_time_mean'] = sum(results['mcts_avg_time']) / len(results['mcts_avg_time'])
    results['baseline_avg_time_mean'] = sum(results['baseline_avg_time']) / len(results['baseline_avg_time'])
    results['avg_steps_mean'] = sum(results['avg_steps']) / len(results['avg_steps'])
    results['mcts_win_rate'] = results['mcts_wins'] / num_games * 100
    
    return results


def experiment_mcts_with_vs_without_llm(
    model: str = "gpt-4o",
    simulations: int = 30,
    num_games: int = 5,
):
    """
    实验2: MCTS+LLM vs MCTS(纯随机模拟)
    """
    print("\n" + "=" * 70)
    print("实验2: MCTS+LLM vs MCTS(随机模拟)")
    print("=" * 70)
    print(f"模型: {model}")
    print(f"MCTS模拟次数: {simulations}")
    print(f"对局数: {num_games}")
    print()
    
    results = {
        'mcts_llm_wins': 0,
        'mcts_random_wins': 0,
        'draws': 0,
        'mcts_llm_avg_time': [],
        'mcts_random_avg_time': [],
        'avg_steps': [],
    }
    
    for game_num in range(num_games):
        print(f"\n{'#'*70}")
        print(f"对局 {game_num + 1}/{num_games}")
        print(f"{'#'*70}")
        
        if game_num % 2 == 0:
            print("MCTS+LLM 执黑，MCTS(随机) 执白")
            agent_black = MCTSLLMAgent(
                name="MCTS+LLM",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=True,
            )
            agent_white = MCTSLLMAgent(
                name="MCTS-Random",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=False,
            )
            llm_is_black = True
        else:
            print("MCTS(随机) 执黑，MCTS+LLM 执白")
            agent_black = MCTSLLMAgent(
                name="MCTS-Random",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=False,
            )
            agent_white = MCTSLLMAgent(
                name="MCTS+LLM",
                model=model,
                api_url=API_URL,
                api_key=API_KEY,
                simulations=simulations,
                use_llm_eval=True,
            )
            llm_is_black = False
        
        stats = play_game_with_stats(agent_black, agent_white, max_steps=100, save_images=False)
        
        if stats['winner'] == "Black":
            winner_is_llm = llm_is_black
        elif stats['winner'] == "White":
            winner_is_llm = not llm_is_black
        else:
            winner_is_llm = None
        
        if winner_is_llm:
            results['mcts_llm_wins'] += 1
            print(f"✓ MCTS+LLM 获胜")
        elif winner_is_llm is False:
            results['mcts_random_wins'] += 1
            print(f"✗ MCTS(随机) 获胜")
        else:
            results['draws'] += 1
            print(f"○ 平局")
        
        if llm_is_black:
            llm_time = stats['agent_a_time'] / stats['agent_a_moves'] if stats['agent_a_moves'] > 0 else 0
            random_time = stats['agent_b_time'] / stats['agent_b_moves'] if stats['agent_b_moves'] > 0 else 0
        else:
            llm_time = stats['agent_b_time'] / stats['agent_b_moves'] if stats['agent_b_moves'] > 0 else 0
            random_time = stats['agent_a_time'] / stats['agent_a_moves'] if stats['agent_a_moves'] > 0 else 0
        
        results['mcts_llm_avg_time'].append(llm_time)
        results['mcts_random_avg_time'].append(random_time)
        results['avg_steps'].append(stats['total_steps'])
        
        print(f"  步数: {stats['total_steps']}")
        print(f"  MCTS+LLM平均用时/步: {llm_time:.2f}秒")
        print(f"  MCTS(随机)平均用时/步: {random_time:.2f}秒")
    
    results['mcts_llm_avg_time_mean'] = sum(results['mcts_llm_avg_time']) / len(results['mcts_llm_avg_time'])
    results['mcts_random_avg_time_mean'] = sum(results['mcts_random_avg_time']) / len(results['mcts_random_avg_time'])
    results['avg_steps_mean'] = sum(results['avg_steps']) / len(results['avg_steps'])
    results['mcts_llm_win_rate'] = results['mcts_llm_wins'] / num_games * 100
    
    return results


def print_experiment_report(exp1_results, exp2_results):
    """打印实验报告"""
    print("\n" + "=" * 70)
    print("实验数据汇总报告")
    print("=" * 70)
    
    print("\n【实验1】MCTS+LLM vs Baseline LLM")
    print("-" * 70)
    print(f"MCTS+LLM 胜: {exp1_results['mcts_wins']} 局")
    print(f"Baseline 胜: {exp1_results['baseline_wins']} 局")
    print(f"平局: {exp1_results['draws']} 局")
    print(f"MCTS+LLM 胜率: {exp1_results['mcts_win_rate']:.1f}%")
    print(f"平均对局步数: {exp1_results['avg_steps_mean']:.1f} 步")
    print(f"MCTS+LLM 平均用时/步: {exp1_results['mcts_avg_time_mean']:.2f} 秒")
    print(f"Baseline 平均用时/步: {exp1_results['baseline_avg_time_mean']:.2f} 秒")
    
    print("\n【实验2】MCTS+LLM vs MCTS(随机模拟)")
    print("-" * 70)
    print(f"MCTS+LLM 胜: {exp2_results['mcts_llm_wins']} 局")
    print(f"MCTS(随机) 胜: {exp2_results['mcts_random_wins']} 局")
    print(f"平局: {exp2_results['draws']} 局")
    print(f"MCTS+LLM 胜率: {exp2_results['mcts_llm_win_rate']:.1f}%")
    print(f"平均对局步数: {exp2_results['avg_steps_mean']:.1f} 步")
    print(f"MCTS+LLM 平均用时/步: {exp2_results['mcts_llm_avg_time_mean']:.2f} 秒")
    print(f"MCTS(随机) 平均用时/步: {exp2_results['mcts_random_avg_time_mean']:.2f} 秒")
    
    print("\n" + "=" * 70)
    
    # 保存到JSON文件
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment1': exp1_results,
        'experiment2': exp2_results,
    }
    
    filename = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验报告已保存到: {filename}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行MCTS+LLM实验并收集数据")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM模型")
    parser.add_argument("--simulations", type=int, default=30, help="MCTS模拟次数")
    parser.add_argument("--num-games", type=int, default=3, help="每个实验的对局数")
    parser.add_argument("--exp1-only", action="store_true", help="只运行实验1")
    parser.add_argument("--exp2-only", action="store_true", help="只运行实验2")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("MCTS+LLM 五子棋实验数据收集")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"MCTS模拟次数: {args.simulations}")
    print(f"每个实验对局数: {args.num_games}")
    print("=" * 70)
    
    exp1_results = None
    exp2_results = None
    
    if not args.exp2_only:
        exp1_results = experiment_mcts_vs_baseline(
            model=args.model,
            simulations=args.simulations,
            num_games=args.num_games,
        )
    
    if not args.exp1_only:
        exp2_results = experiment_mcts_with_vs_without_llm(
            model=args.model,
            simulations=args.simulations,
            num_games=args.num_games,
        )
    
    if exp1_results and exp2_results:
        print_experiment_report(exp1_results, exp2_results)
    elif exp1_results:
        print("\n实验1完成！")
    elif exp2_results:
        print("\n实验2完成！")

