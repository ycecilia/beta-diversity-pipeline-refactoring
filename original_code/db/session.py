"""
Stub database session for challenge.
"""


class MockSession:
    """Mock database session that doesn't actually connect to a database."""
    
    def query(self, model):
        """Mock query method."""
        return MockQuery(model)
    
    def add(self, obj):
        """Mock add method."""
        print(f"[MOCK] Adding {obj.__class__.__name__} to session")
        pass
    
    def commit(self):
        """Mock commit method."""
        print("[MOCK] Committing session")
        pass
    
    def close(self):
        """Mock close method."""
        print("[MOCK] Closing session")
        pass


class MockQuery:
    """Mock query object."""
    
    def __init__(self, model):
        self.model = model
    
    def filter(self, *args):
        """Mock filter method."""
        return self
    
    def first(self):
        """Mock first method - returns a mock report."""
        if hasattr(self.model, '__name__') and self.model.__name__ == 'MockReport':
            return self.model()
        return None
    
    def update(self, values, synchronize_session=None):
        """Mock update method."""
        print(f"[MOCK] Updating {self.model.__name__} with {values}")
        pass


def start_db_session(namespace: str = "staging"):
    """
    Mock function to start a database session.
    
    Returns a mock session that allows the code to run without a real database.
    """
    print(f"[MOCK] Starting database session for namespace: {namespace}")
    return MockSession()