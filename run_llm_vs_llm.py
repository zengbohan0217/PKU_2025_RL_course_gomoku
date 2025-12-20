import os
import datetime

from gomoku_env import GomokuEnv
from methods.llm_agent import LLMGomokuAgent
from methods import api


API_URL = api.DEFAULT_API_URL
API_KEY = api.DEFAULT_API_KEY


def create_game_dir(base_dir: str = "game_records") -> str:
    """
    为每一盘对局创建一个独立的目录，用于保存每一步的棋盘图片。
    目录格式类似：game_records/game_20251220_230512
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    game_dir = os.path.join(base_dir, f"game_{ts}")
    os.makedirs(game_dir, exist_ok=True)
    return game_dir


def save_board_image(env: GomokuEnv, game_dir: str, step: int) -> str:
    """
    将当前棋盘渲染为图片，并保存到指定对局目录中。
    文件名格式：step_000.png, step_001.png, ...
    返回保存路径，便于需要时引用。
    """
    img = env.render_image()
    path = os.path.join(game_dir, f"step_{step:03d}.png")
    img.save(path)
    return path


def save_final_image(env: GomokuEnv, game_dir: str) -> str:
    """
    将终局棋盘单独保存为 final.png，便于快速查看最后局面。
    """
    img = env.render_image()
    path = os.path.join(game_dir, "final.png")
    img.save(path)
    return path


def play_llm_vs_llm(
    model_a: str,
    model_b: str,
    use_vlm_a: bool = False,
    use_vlm_b: bool = False,
    max_steps: int = 50,
):
    """
    让两个 LLM / VLM 之间进行完整对弈，并且记录每一步的棋盘图片。

    参数：
    - model_a, model_b: 两个基座模型名称，例如 "gpt-4", "gpt-4-vision" 等
    - use_vlm_a, use_vlm_b: 是否为对应 agent 启用图像输入（VLM）
    - max_steps: 最多落子步数，上限保护
    """
    env = GomokuEnv()
    env.reset()

    # 为本盘对局创建记录目录
    game_dir = create_game_dir()
    print(f"Game records will be saved to: {game_dir}")

    agent_a = LLMGomokuAgent(
        name="AgentA",
        model=model_a,
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=use_vlm_a,
    )
    agent_b = LLMGomokuAgent(
        name="AgentB",
        model=model_b,
        api_url=API_URL,
        api_key=API_KEY,
        use_vlm=use_vlm_b,
    )

    # 约定：AgentA 执黑先手，AgentB 执白
    is_white_turn = False  # False -> Black(A), True -> White(B)
    step = 0

    # 初始局面也可以保存一张图片（可选）
    save_board_image(env, game_dir, step)

    while step < max_steps:
        print("\n" + "=" * 40)
        print(f"Step {step}")
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
        print(f"\n{agent.name} ({side_name}) is thinking... (use_vlm={agent.use_vlm})")

        # 若该 agent 配置为 VLM，则使用棋盘图像作为输入之一
        use_image = agent.use_vlm
        action = agent.choose_action(env, is_white=is_white, use_image=use_image)
        print(f"{agent.name} ({side_name}) plays at {action}")

        current_white, current_black, reward, done, info = env.step(action, is_white=is_white)
        step += 1
        is_white_turn = not is_white_turn

        # 保存当前步之后的棋盘图片
        save_board_image(env, game_dir, step)

        if done:
            print("\nFinal board:")
            env.render()
            print("\nGame over! Winner:", info.get("winner"))
            # 单独保存一份终局图片
            save_final_image(env, game_dir)
            break

    if env.winner is None:
        print("\nReached max steps without a winner. Treat as draw.")
        env.render()
        # 对于未分胜负的情况，也保存一个 final.png 作为终局
        save_final_image(env, game_dir)


if __name__ == "__main__":
    # 示例 1：两个纯文本 LLM 对弈（不使用图片）
    play_llm_vs_llm(
        model_a="gpt-4o",
        model_b="gpt-5.1",
        use_vlm_a=False,
        use_vlm_b=False,
    )

    # 示例 2：左边文本 LLM，右边 VLM（右边使用棋盘图片 + 文本）
    play_llm_vs_llm(
        model_a="gpt-4o",
        model_b="gpt-4o",
        use_vlm_a=False,
        use_vlm_b=True,
    )
