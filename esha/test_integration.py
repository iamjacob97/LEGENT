#!/usr/bin/env python3
"""
Integration test for ESHAController + ESHAAnalyzer
"""

import sys
import os

# Add LEGENT to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from esha.esha_controller import ESHAController
from esha.esha_analyser import ESHAAnalyzer, run_esha_analysis


def test_basic_integration():
    """Test basic integration between controller and analyzer."""
    print("🧪 Testing ESHAController + ESHAAnalyzer integration...")
    
    # 1. Generate a scene using ESHAController
    print("📋 Step 1: Creating ESHAController...")
    controller = ESHAController()
    
    print("🏗️  Step 2: Generating scene...")
    scene = controller.generate_candidate()
    
    print(f"✅ Scene generated with {len(scene.get('instances', []))} instances")
    print(f"   - Rooms: {len(scene.get('room_polygon', []))}")
    print(f"   - Player: {'present' if scene.get('player') else 'missing'}")
    print(f"   - Agent: {'present' if scene.get('agent') else 'missing'}")
    
    # 2. Analyze the scene (without LLM first)
    print("\n🔍 Step 3: Running ESHA analysis (rule-based fallback)...")
    analyzer = ESHAAnalyzer()  # No API key = fallback mode
    result = analyzer.analyze(scene_dict=scene)
    
    print(f"✅ Analysis complete!")
    print(f"   - Found {len(result.hazards)} hazards")
    print(f"   - Report length: {len(result.report_markdown)} characters")
    print(f"   - Room metrics: {len(result.summary['metrics']['rooms'])} rooms")
    
    # 3. Test one-shot convenience function  
    print("\n🚀 Step 4: Testing convenience function...")
    result2 = run_esha_analysis(scene_dict=scene)
    print(f"✅ Convenience function works! Found {len(result2.hazards)} hazards")
    
    # 4. Print sample output
    print("\n📊 Sample hazards found:")
    for i, hazard in enumerate(result.hazards[:3]):  # Show first 3
        print(f"   {i+1}. {hazard.item} - {hazard.consequence}")
    
    print(f"\n📝 Report preview (first 500 chars):")
    print(result.report_markdown[:500] + "..." if len(result.report_markdown) > 500 else result.report_markdown)
    
    return True


def test_edge_cases():
    """Test analyzer with various ESHAController configurations."""
    print("\n🧪 Testing edge cases...")
    
    # Test with different room types
    from esha.esha_controller import SceneConfig
    
    room_types = ["Kitchen", "Bathroom", "Bedroom", "LivingRoom"]
    
    for room_type in room_types:
        print(f"\n🏠 Testing {room_type}...")
        config = SceneConfig(
            room_type=room_type,
            dims=(4, 4),  # Small room
            include_other_items=True,
            items={"orange": 1, "table": 1}
        )
        
        controller = ESHAController(scene_config=config)
        scene = controller.generate_candidate()
        
        analyzer = ESHAAnalyzer()
        result = analyzer.analyze(scene_dict=scene)
        
        print(f"   ✅ {room_type}: {len(result.hazards)} hazards, {len(result.summary['metrics']['rooms'])} rooms")
    
    return True


if __name__ == "__main__":
    try:
        print("🚀 Starting ESHA Integration Tests...\n")
        
        # Run basic integration test
        test_basic_integration()
        
        # Run edge case tests
        test_edge_cases()
        
        print("\n🎉 All tests passed! Integration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
