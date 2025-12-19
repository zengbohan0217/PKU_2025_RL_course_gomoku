from gomoku_env import GomokuEnv


env = GomokuEnv()
# 示例落子
env.step((7, 7), is_white=False)  # 黑棋落子
env.step((7, 8), is_white=True)   # 白棋落子
env.step((8, 7), is_white=False)  # 黑棋落子
env.step((8, 8), is_white=True)   # 白棋落子
env.step((9, 7), is_white=False)  # 黑棋落子
env.step((9, 8), is_white=True)   # 白棋落子
env.step((10, 7), is_white=False) # 黑棋落子
env.step((10, 8), is_white=True)  # 白棋落子
env.step((11, 7), is_white=False) # 黑棋落子，黑棋获胜

image = env.render_image()
image.show()  # 显示图片
