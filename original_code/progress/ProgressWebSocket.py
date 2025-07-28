"""
Stub for WebSocket progress reporting.
"""


class ProgressWebSocket:
    """Mock WebSocket class for progress reporting."""
    
    def __init__(self, report_id: str):
        self.report_id = report_id
        print(f"[MOCK] Created WebSocket for report: {report_id}")
    
    def send_update(self, status: str, progress: float, message: str):
        """Mock method to send progress updates."""
        print(f"[MOCK] Progress {progress*100:.1f}%: {status} - {message}")
    
    def close(self):
        """Mock method to close WebSocket connection."""
        print(f"[MOCK] Closing WebSocket for report: {self.report_id}")