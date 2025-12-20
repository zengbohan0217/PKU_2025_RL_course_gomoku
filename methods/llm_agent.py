import re
from typing import Tuple, Optional

from gomoku_env import GomokuEnv
from . import api
from . import prompts


def board_to_text(env: GomokuEnv) -> str:
    """
    将当前棋盘 + 最近若干步历史，转成易读文本。
    B = black, W = white, . = empty
    """
    lines = []
    lines.append("Current board (15x15):")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (x, y) in env.current_black:
                row.append("B")
            elif (x, y) in env.current_white:
                row.append("W")
            else:
                row.append(".")
        lines.append("".join(row))

    if env.history:
        lines.append("\nLast moves (most recent last):")
        for x, y, is_white in env.history[-10:]:
            c = "W" if is_white else "B"
            lines.append(f"{c} at ({x}, {y})")
    return "\n".join(lines)


def parse_move(text: str) -> Optional[Tuple[int, int]]:
    """
    从模型输出中提取 'x,y' 两个整数坐标。
    """
    m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", text)
    if not m:
        return None
    x = int(m.group(1))
    y = int(m.group(2))
    return x, y


class LLMGomokuAgent:
    """
    一个基于 LLM 的五子棋 Agent。
    - 支持纯文本 LLM（只用 prompt）
    - 支持 VLM（传棋盘截图 + prompt）
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_url: str,
        api_key: str,
        use_vlm: Optional[bool] = None,
    ):
        """
        name: 例如 'AgentA'，用于日志打印
        model: 基座模型名称（如 'gpt-4', 'gpt-4-vision', 'gemini-xxx'）
        api_url: OpenAI 兼容 /v1 地址
        api_key: 接口 Key
        use_vlm: 显式指定是否用图像；若为 None，则根据模型名自动判断
        """
        self.name = name
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.use_vlm = api.is_vlm_model(model) if use_vlm is None else use_vlm

    def _build_messages(
        self,
        env: GomokuEnv,
        is_white: bool,
        image_path: Optional[str] = None,
    ):
        """
        构造 /chat/completions 所需的 messages。
        - 若 self.use_vlm 且 image_path 非空，则使用 text + image 混合格式；
        - 否则只用纯文本。
        """
        color = "White" if is_white else "Black"
        board_str = board_to_text(env)
        user_prompt = prompts.build_user_prompt_from_board(board_str, color)

        if self.use_vlm and image_path is not None:
            # VLM: system + user(content=[text, image_url])
            system_prompt = prompts.VLM_SYSTEM_PROMPT
            content = api.build_image_message_content(user_prompt, image_path)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            # 纯文本 LLM
            system_prompt = prompts.TEXT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return messages

    def choose_action(
        self,
        env: GomokuEnv,
        is_white: bool,
        max_retry: int = 3,
        use_image: bool = False,
        image_path: str = "current_board.png",
    ) -> Tuple[int, int]:
        """
        主接口：给定当前 env 状态和执子方，返回一个合法坐标 (x,y)。

        - 如果 use_image=True 且 agent.use_vlm=True，
          会将当前棋盘渲染为图片（保存到 image_path）并随同 prompt 一起发给模型。
        - 如果 use_image=False 或 agent.use_vlm=False，
          则走纯文本模式，仅发送棋盘文本描述。
        """
        actual_image_path: Optional[str] = None
        if use_image and self.use_vlm:
            img = env.render_image()
            img.save(image_path)
            actual_image_path = image_path

        for _ in range(max_retry):
            messages = self._build_messages(env, is_white, image_path=actual_image_path)
            raw = api.call_chat_completions(
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=32,
            )
            move = parse_move(raw)
            if move is None:
                continue
            if env.check_step(move):
                return move

        # 兜底策略：从左上到右下找第一个空位
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.current_white and (x, y) not in env.current_black:
                    return x, y

        # 理论上不会到这里
        raise RuntimeError("No legal move found.")
