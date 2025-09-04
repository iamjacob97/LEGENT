# Hazalyser Package - Scene Generation and Analysis Tools
# 
# This package provides tools for generating, analyzing, and controlling 
# single-room scenes in LEGENT with a focus on safety and accessibility.

# Core Controller and Configuration
from .controller import Controller
from .generator import SceneConfig, SceneGenerator

# House Generation
from .house import Room, RoomSpec, House, generate_house_structure

# Analysis and Safety Tools
from .prompter import Prompter, run_llm_analysis, HazardEntry, Result

# Utility Functions
from .helpers import split_items_into_receptacles_and_objects

# Object Database
from .objects import ObjectDB, get_default_object_db

__all__ = [
    # Core Classes
    "Controller",          # Main controller for scene generation and interaction
    "SceneGenerator",      # Extended HouseGenerator with ESHA-specific functionality
    
    # Configuration and Specification
    "SceneConfig",         # Configuration for scene generation (dimensions, items, etc.)
    "Room",               # Individual room representation
    "RoomSpec",           # Room specification for scene generation
    "House",              # House structure representation
    
    # Analysis Tools
    "Prompter",           # Safety and accessibility analysis
    "run_llm_analysis",   # Convenience function for running analysis
    "HazardEntry",        # Individual hazard entry data structure
    "Result",             # Analysis result container
    
    # Utility Functions
    "split_items_into_receptacles_and_objects",  # Item categorization helper
    "generate_house_structure",                  # House structure generation
    
    # Object Database
    "ObjectDB",           # Object database for scene generation
    "get_default_object_db",  # Get default object database instance
    
]

# Version information
__version__ = "1.0.0"

# Package description
__doc__ = """
Hazalyser Package - Enhanced Scene Generation and Analysis

A comprehensive toolkit for generating, analyzing, and controlling single-room 
scenes in LEGENT with a focus on safety, accessibility, and user control.

Key Features:
- Single-room scene generation with user-specified items
- Safety and accessibility analysis using LLM-based hazard identification
- Interactive scene control and modification
- Flexible room configuration and sizing
- Integration with LEGENT's procedural generation system

Usage:
    from hazalyser import Controller, SceneConfig, Prompter
    
    # Create a scene configuration
    config = SceneConfig(
        room_spec=RoomSpec(
            room_spec_id="test",
            spec=Room(room_id=1, room_type="Kitchen")
        ),
        dims=(4, 4),
        items={"table": 1, "chair": 2}
    )
    
    # Generate and control scenes
    controller = Controller()
    scene = controller.generate_candidate(config)
    
    # Analyze for hazards
    analyzer = Prompter()
    result = analyzer.analyze(scene)
"""


