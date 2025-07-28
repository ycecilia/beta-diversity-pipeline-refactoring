"""
Stub database schema for challenge.

This provides mock database models that allow the original code to run
without requiring a real database connection.
"""
from datetime import datetime
from typing import Optional, List


class MockReport:
    """Mock Report model for challenge."""
    
    def __init__(self):
        self.id = "test_report_123"
        self.project_id = "proj_123"
        self.marker = "test_primer"
        self.taxonomicRank = "species"
        self.sites = ["site_a", "site_b"]
        self.tags = []
        self.firstDate = "2023-01-01"
        self.lastDate = "2023-12-31"
        self.environmentalParameter = "bio01"
        self.countThreshold = 100
        self.confidenceLevel = 0.8
        self.filterThreshold = 10
        self.betaDiversity = "braycurtis"
        self.alphaDiversity = "shannon"
        self.speciesList = "None"


class MockComputeLog:
    """Mock ComputeLog model for challenge."""
    
    def __init__(self, project_id: str, description: str, operation: str, 
                 executedAt: datetime, duration: int, cores: int, memory: int):
        self.project_id = project_id
        self.description = description
        self.operation = operation
        self.executedAt = executedAt
        self.duration = duration
        self.cores = cores
        self.memory = memory


# Provide the same interface as the real models
Report = MockReport
ComputeLog = MockComputeLog