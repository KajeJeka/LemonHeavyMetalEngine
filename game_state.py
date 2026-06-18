STATE_MENU    = "MENU"
STATE_PLAYING = "PLAYING"


class GameStateManager:
    def __init__(self):
        self.state       = STATE_MENU
        self.map_mode    = "DEFAULT"   # "DEFAULT" | "INFINITE"
        self.map_seed: int | None = None

        # Menu cursor: 0 = Play Default, 1 = Play Infinite
        self.cursor      = 0
        self._options    = ["PLAY DEFAULT", "PLAY INFINITE"]

    @property
    def in_menu(self) -> bool:
        return self.state == STATE_MENU

    @property
    def in_game(self) -> bool:
        return self.state == STATE_PLAYING

    def select_up(self):
        self.cursor = (self.cursor - 1) % len(self._options)

    def select_down(self):
        self.cursor = (self.cursor + 1) % len(self._options)

    def confirm(self) -> str:
        """Returns 'DEFAULT' or 'INFINITE' and transitions to PLAYING."""
        choice = "DEFAULT" if self.cursor == 0 else "INFINITE"
        self.map_mode = choice
        self.state    = STATE_PLAYING
        return choice

    def return_to_menu(self):
        self.state = STATE_MENU

    @property
    def options(self) -> list[str]:
        return self._options
