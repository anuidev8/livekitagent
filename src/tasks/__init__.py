"""Phase specialist tasks for the Huella Digital supervisor agent."""

from tasks.analysis import AnalysisTask
from tasks.attract import AttractTask
from tasks.base import PhaseResult
from tasks.closing import ClosingTask
from tasks.detail import DetailTask
from tasks.intro import IntroTask
from tasks.recommendations import RecommendationsTask
from tasks.welcome import WelcomeTask

__all__ = [
    "AnalysisTask",
    "AttractTask",
    "ClosingTask",
    "DetailTask",
    "IntroTask",
    "PhaseResult",
    "RecommendationsTask",
    "WelcomeTask",
]
