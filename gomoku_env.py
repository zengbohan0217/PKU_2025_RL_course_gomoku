import os
from PIL import Image, ImageDraw, ImageFont

class GomokuEnv:
    def __init__(self, height=15, width=15):
        self.height = height
        self.width = width
        # 使用集合(set)来存储位置，查找速度比列表(list)快，复杂度为O(1)
        # 位置格式建议使用元组 (x, y)
        self.current_white = set() 
        self.current_black = set() 
        self.history = []  # 记录历史动作，格式可以是 [(x, y, is_white), ...]
        self.winner = None # 记录赢家，None表示未分胜负

        self.cell_size = 40  # 每个格子的像素大小
        self.margin = 30     # 棋盘边缘留白
        self.radius = 16     # 棋子半径

    def check_step(self, action):
        """
        检查动作是否合法
        :param action: 元组 (x, y)，表示落子坐标
        :return: bool, 如果合法返回 True，否则 False
        """
        x, y = action
        
        # 1. 检查坐标是否在棋盘范围内
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # 2. 检查该位置是否已经有棋子
        if (x, y) in self.current_white or (x, y) in self.current_black:
            return False
            
        # 3. 检查游戏是否已经结束
        if self.winner is not None:
            return False

        return True

    def step(self, action, is_white):
        """
        执行一步落子
        :param action: 元组 (x, y)
        :param is_white: bool, True表示白棋，False表示黑棋
        :return: tuple (observation, reward, done, info) - 模仿 Gym 接口
        """
        if not self.check_step(action):
            raise ValueError(f"Invalid action: {action}. Position might be occupied or out of bounds.")

        # 更新棋盘状态
        if is_white:
            self.current_white.add(action)
        else:
            self.current_black.add(action)
        
        # 记录历史
        self.history.append((action[0], action[1], is_white))

        # 检查是否获胜
        # 注意：这里我们只检查当前落子的这一方是否获胜
        current_player_positions = self.current_white if is_white else self.current_black
        won = self.check_winner(action, current_player_positions)
        
        done = False
        info = {}
        reward = 0

        if won:
            self.winner = "White" if is_white else "Black"
            done = True
            reward = 1 # 赢家获得奖励
            info['winner'] = self.winner
        elif len(self.current_white) + len(self.current_black) == self.height * self.width:
            # 棋盘满了，平局
            done = True
            reward = 0
            info['winner'] = "Draw"
            self.winner = "Draw"

        return (self.current_white, self.current_black), reward, done, info

    def check_winner(self, last_action, current_player_positions):
        """
        检查当前棋盘状态是否有赢家
        优化策略：只需要检查最后落子的位置周围是否连成5子
        
        :param last_action: 最后落子的坐标 (x, y)
        :param current_player_positions: 当前落子方的所有棋子位置集合
        """
        if last_action is None:
            return False
            
        x, y = last_action
        
        # 四个方向：横向，纵向，主对角线(\)，副对角线(/)
        directions = [
            [(1, 0), (-1, 0)],  # 横向
            [(0, 1), (0, -1)],  # 纵向
            [(1, 1), (-1, -1)], # 主对角线
            [(1, -1), (-1, 1)]  # 副对角线
        ]

        for axis in directions:
            count = 1 # 当前落子算1个
            # 向两个相反的方向延伸搜索
            for dx, dy in axis:
                curr_x, curr_y = x + dx, y + dy
                while (curr_x, curr_y) in current_player_positions:
                    count += 1
                    curr_x += dx
                    curr_y += dy
            
            # 标准五子棋胜利条件：连成5个或以上
            if count >= 5:
                return True

        return False

    def render(self):
        """
        可视化当前棋盘状态 (控制台打印)
        """
        # 清屏 (可选，视运行环境而定)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n  " + " ".join([f"{i:2}" for i in range(self.width)])) # 打印列号
        
        for y in range(self.height):
            row_str = f"{y:2} " # 打印行号
            for x in range(self.width):
                if (x, y) in self.current_black:
                    row_str += "●  " # 黑棋
                elif (x, y) in self.current_white:
                    row_str += "○  " # 白棋
                else:
                    row_str += "+  " # 空位
            print(row_str)
        
        if self.winner:
            print(f"\nGame Over! Winner: {self.winner}")
        else:
            last_move = self.history[-1] if self.history else None
            if last_move:
                player = "White" if last_move[2] else "Black"
                print(f"\nLast move: {player} at ({last_move[0]}, {last_move[1]})")
    
    def render_image(self):
        """
        将当前棋盘状态保存为 RGB 图片
        """
        # 1. 计算图片总尺寸
        img_width = self.width * self.cell_size + 2 * self.margin
        img_height = self.height * self.cell_size + 2 * self.margin
        
        # 2. 创建画布 (背景色: 类似木纹的颜色或者纯色，这里用淡黄色)
        board_color = (235, 200, 120) 
        image = Image.new("RGB", (img_width, img_height), board_color)
        draw = ImageDraw.Draw(image)

        # 3. 绘制网格线
        line_color = (0, 0, 0)
        # 竖线
        for i in range(self.width):
            x = self.margin + i * self.cell_size
            start = (x, self.margin)
            end = (x, img_height - self.margin - self.cell_size) # 注意：通常棋盘线不画满边缘，这里为了简单画满格子区域
            # 修正：让线画到最后一个交叉点
            end = (x, self.margin + (self.height - 1) * self.cell_size)
            draw.line([start, end], fill=line_color, width=1)
        
        # 横线
        for i in range(self.height):
            y = self.margin + i * self.cell_size
            start = (self.margin, y)
            end = (self.margin + (self.width - 1) * self.cell_size, y)
            draw.line([start, end], fill=line_color, width=1)

        # 4. 绘制棋子
        def draw_piece(x_grid, y_grid, color):
            # 计算圆心坐标
            center_x = self.margin + x_grid * self.cell_size
            center_y = self.margin + y_grid * self.cell_size
            # 计算左上角和右下角坐标
            left_up = (center_x - self.radius, center_y - self.radius)
            right_down = (center_x + self.radius, center_y + self.radius)
            
            draw.ellipse([left_up, right_down], fill=color, outline=(0,0,0))
            
            # 增加一点高光让棋子更有立体感（可选）
            highlight_offset = self.radius // 3
            highlight_r = self.radius // 4
            h_left_up = (center_x - highlight_offset - highlight_r, center_y - highlight_offset - highlight_r)
            h_right_down = (center_x - highlight_offset + highlight_r, center_y - highlight_offset + highlight_r)
            # 只有黑棋加明显高光，白棋本身就很亮
            if color == (0, 0, 0):
                draw.ellipse([h_left_up, h_right_down], fill=(100, 100, 100))

        # 画黑棋
        for x, y in self.current_black:
            draw_piece(x, y, (0, 0, 0)) # 黑色

        # 画白棋
        for x, y in self.current_white:
            draw_piece(x, y, (255, 255, 255)) # 白色

        # 5. 标记最后一步 (红色小点)
        if self.history:
            last_x, last_y, _ = self.history[-1]
            center_x = self.margin + last_x * self.cell_size
            center_y = self.margin + last_y * self.cell_size
            r = 4
            draw.ellipse([(center_x-r, center_y-r), (center_x+r, center_y+r)], fill=(255, 0, 0))

        # 6. 保存图片
        return image

    def reset(self):
        """
        重置环境
        """
        self.current_white.clear()
        self.current_black.clear()
        self.history = []
        self.winner = None
        return (self.current_white, self.current_black)
