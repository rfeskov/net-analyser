import json
import os
from typing import Set

class Storage:
    def __init__(self, filename: str = "subscribers.json"):
        self.filename = filename
        self.subscribers: Set[int] = set()
        self._load()

    def _load(self):
        """Load data from file if it exists"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.subscribers = set(data.get('subscribers', []))
            except Exception as e:
                print(f"Error loading storage: {e}")

    def _save(self):
        """Save data to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'subscribers': list(self.subscribers)
                }, f)
        except Exception as e:
            print(f"Error saving storage: {e}")

    def add_subscriber(self, user_id: int) -> bool:
        """Add subscriber (admin only)"""
        if user_id not in self.subscribers:
            self.subscribers.add(user_id)
            self._save()
            return True
        return False

    def remove_subscriber(self, user_id: int) -> bool:
        """Remove a subscriber"""
        if user_id in self.subscribers:
            self.subscribers.remove(user_id)
            self._save()
            return True
        return False

    def get_subscribers(self) -> Set[int]:
        """Get all subscribers"""
        return self.subscribers

    def is_subscriber(self, user_id: int) -> bool:
        """Check if user is a subscriber"""
        return user_id in self.subscribers 