from gomoku_env import GomokuEnv


env = GomokuEnv()
# 示例落子
## -> 获取棋子状态送给LLM进行分析
current_white, current_black, reward, done, info = env.step((7, 7), is_white=False)  # 黑棋落子
current_white, current_black, reward, done, info = env.step((7, 8), is_white=True)   # 白棋落子
current_white, current_black, reward, done, info = env.step((8, 7), is_white=False)  # 黑棋落子
current_white, current_black, reward, done, info = env.step((8, 8), is_white=True)   # 白棋落子
current_white, current_black, reward, done, info = env.step((9, 7), is_white=False)  # 黑棋落子
current_white, current_black, reward, done, info = env.step((9, 8), is_white=True)   # 白棋落子
current_white, current_black, reward, done, info = env.step((10, 7), is_white=False) # 黑棋落子
current_white, current_black, reward, done, info = env.step((10, 8), is_white=True)  # 白棋落子
current_white, current_black, reward, done, info = env.step((11, 7), is_white=False) # 黑棋落子，黑棋获胜
current_white, current_black, reward, done, info = env.step((11, 8), is_white=True)  # 白棋落子
print("Current White Positions:", current_white)
print("Current Black Positions:", current_black)

image = env.render_image() ## -> 获取图片送给VLM进行分析
## 可以设置成，如果博弈50步还没分出胜负，就保存当前棋盘图片进行分析
image.save("gomoku_board.png")
