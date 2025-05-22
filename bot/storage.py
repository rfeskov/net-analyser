import json
import os
from typing import Set

class Storage:
    def __init__(self, filename: str = "subscribers.json"):
        self.filename = filename
        self.pending_subscribers: Set[int] = set()
        self.subscribers: Set[int] = set()
        self._load()

    def _load(self):
        """Load data from file if it exists"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.subscribers = set(data.get('subscribers', []))
                    self.pending_subscribers = set(data.get('pending', []))
            except Exception as e:
                print(f"Error loading storage: {e}")

    def _save(self):
        """Save data to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'subscribers': list(self.subscribers),
                    'pending': list(self.pending_subscribers)
                }, f)
        except Exception as e:
            print(f"Error saving storage: {e}")

    def add_pending(self, user_id: int) -> bool:
        """Add user to pending subscribers"""
        if user_id not in self.subscribers and user_id not in self.pending_subscribers:
            self.pending_subscribers.add(user_id)
            self._save()
            return True
        return False

    def approve_subscriber(self, user_id: int) -> bool:
        """Approve a pending subscriber"""
        if user_id in self.pending_subscribers:
            self.pending_subscribers.remove(user_id)
            self.subscribers.add(user_id)
            self._save()
            return True
        return False

    def reject_subscriber(self, user_id: int) -> bool:
        """Reject a pending subscriber"""
        if user_id in self.pending_subscribers:
            self.pending_subscribers.remove(user_id)
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
        """Get all approved subscribers"""
        return self.subscribers

    def get_pending_subscribers(self) -> Set[int]:
        """Get all pending subscribers"""
        return self.pending_subscribers 