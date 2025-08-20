# ESHA Package - Scene Generation and Analysis Tools
# 
# This package provides tools for generating, analyzing, and controlling 
# single-room scenes in LEGENT with a focus on safety and accessibility.

# Core Controller and Configuration
from .esha_controller import ESHAController
from .esha_generator import SceneConfig, ESHARoom, ESHARoomSpec, ESHAGenerator

# House Generation
from .esha_house import generate_house_structure

# Analysis and Safety Tools
from .esha_analyser import ESHAAnalyzer, run_esha_analysis

# Utility Functions
from .esha_helpers import split_items_into_receptacles_and_objects

__all__ = [
    # Core Classes
    "ESHAController",      # Main controller for scene generation and interaction
    "ESHAGenerator",       # Extended HouseGenerator with ESHA-specific functionality
    
    # Configuration and Specification
    "SceneConfig",         # Configuration for scene generation (dimensions, items, etc.)
    "ESHARoom",           # Individual room representation
    "ESHARoomSpec",       # Room specification for scene generation
    
    # Analysis Tools
    "ESHAAnalyzer",       # Safety and accessibility analysis
    "run_esha_analysis",  # Convenience function for running analysis
    
    # Utility Functions
    "split_items_into_receptacles_and_objects",  # Item categorization helper
    "generate_house_structure",                  # House structure generation
    
]

# Version information
__version__ = "1.0.0"

# Package description
__doc__ = """
ESHA Package - Enhanced Scene Generation and Analysis

A comprehensive toolkit for generating, analyzing, and controlling single-room 
scenes in LEGENT with a focus on safety, accessibility, and user control.

Key Features:
- Single-room scene generation with user-specified items
- Safety and accessibility analysis
- Interactive scene control and modification
- Flexible room configuration and sizing
- Integration with LEGENT's procedural generation system

Usage:
    from esha import ESHAController, SceneConfig
    
    # Create a scene configuration
    config = SceneConfig(
        room_type="Kitchen",
        dims=(4, 4),
        items={"table": 1, "chair": 2}
    )
    
    # Generate and control scenes
    controller = ESHAController(scene_config=config)
    scene = controller.generate_candidate()
"""


