import curses
import random
import time


class SnakeGame:
    def __init__(self):
        self.window = curses.initscr()
        self.window.timeout(100)
        self.window.keypad(1)
        curses.curs_set(0)
        self.height, self.width = self.window.getmaxyx()
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = curses.KEY_RIGHT
        self.apple = self._generate_apple()
        self.score = 0
        self.game_over = False

    def _generate_apple(self):
        while True:
            apple = (random.randint(1, self.height - 2),
                     random.randint(1, self.width - 2))
            if apple not in self.snake:
                return apple

    def _update_direction(self, key):
        if key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
            if (key == curses.KEY_UP and self.direction != curses.KEY_DOWN or
                key == curses.KEY_DOWN and self.direction != curses.KEY_UP or
                key == curses.KEY_LEFT and self.direction != curses.KEY_RIGHT or
                    key == curses.KEY_RIGHT and self.direction != curses.KEY_LEFT):
                self.direction = key

    def _move_snake(self):
        head = self.snake[0]
        new_head = (head[0] + (self.direction == curses.KEY_DOWN and 1) + (self.direction == curses.KEY_UP and -1),
                    head[1] + (self.direction == curses.KEY_LEFT and -1) + (self.direction == curses.KEY_RIGHT and 1))
        self.snake.insert(0, new_head)

    def _check_collisions(self):
        head = self.snake[0]
        if (head[0] in [0, self.height - 1] or head[1] in [0, self.width - 1] or head in self.snake[1:]):
            self.game_over = True
            return True
        return False

    def _check_apple(self):
        if self.snake[0] == self.apple:
            self.score += 1
            self.apple = self._generate_apple()
        else:
            self.snake.pop()

    def _draw_elements(self):
        self.window.clear()
        self.window.border(0)
        self.window.addch(self.apple[0], self.apple[1], curses.ACS_DIAMOND)
        for segment in self.snake:
            self.window.addch(segment[0], segment[1], '#')
        self.window.addstr(0, 2, f'Score: {self.score} ')
        self.window.refresh()

    def play(self):
        while not self.game_over:
            self._draw_elements()
            key = self.window.getch()
            if key == -1:
                key = self.direction
            if key == 27:
                break
            self._update_direction(key)
            self._move_snake()
            if self._check_collisions():
                break
            self._check_apple()

        self.window.clear()
        self.window.border(0)
        game_over_text = f'Game Over! Your score: {self.score}'
        self.window.addstr(self.height // 2, (self.width -
                           len(game_over_text)) // 2, game_over_text)
        self.window.refresh()
        self.window.getch()
        time.sleep(1)
        curses.endwin()


if __name__ == '__main__':
    game = SnakeGame()
    game.play()
