"""
Stub database functions for challenge.
"""


def update_report_status(session, report_id: str, status: str, in_progress: bool, 
                        error_message: str = None, error_type: str = None):
    """
    Mock function to update report status.
    
    In the challenge, this does nothing but allows the code to run.
    """
    print(f"[MOCK] Updating report {report_id} status to {status}")
    if error_message:
        print(f"[MOCK] Error: {error_message}")
    pass