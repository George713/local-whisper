from pynput import keyboard
import time

class KeyboardHandler:
    def __init__(self, hotkey_name, on_toggle_callback):
        self.controller = keyboard.Controller()
        self.hotkey_name = hotkey_name
        self.on_toggle_callback = on_toggle_callback
        self.target_key = self._parse_key(hotkey_name)
        self.listener = None

    def _parse_key(self, name):
        try:
            # Check if it's a special key like Key.caps_lock
            return getattr(keyboard.Key, name.lower())
        except AttributeError:
            # Otherwise treat as a character
            return keyboard.KeyCode.from_char(name)

    def _on_press(self, key):
        if key == self.target_key:
            self.on_toggle_callback()

    def start(self):
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()

    def inject_text(self, text, mode="type"):
        if not text:
            return
            
        if mode == "type":
            # Small delay to ensure focus is back on the target field
            time.sleep(0.1)
            self.controller.type(text)
        elif mode == "clipboard":
            # This would require a clipboard library like pyperclip
            # For now, we stick to typing as it's more universal
            self.controller.type(text)
