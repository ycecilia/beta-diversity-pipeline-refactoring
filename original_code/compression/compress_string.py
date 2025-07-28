"""
Stub for string compression functionality.
"""


def compress_string(data: str) -> bytes:
    """
    Mock function to compress a string.
    
    In the challenge, this just returns the string as bytes without compression.
    """
    print(f"[MOCK] Compressing {len(data)} characters")
    return data.encode('utf-8')