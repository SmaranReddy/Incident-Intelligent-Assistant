class ChatMemory:
    def __init__(self):
        self.history = {}

    def get(self, session_id):
        return self.history.get(session_id, [])

    def add(self, session_id, user, bot):
        if session_id not in self.history:
            self.history[session_id] = []

        self.history[session_id].append({
            "user": user,
            "bot": bot
        })