# =====================================================
# scripts/pattern_optimization/create_custom_params.py - Custom Parameter Template Generator
# =====================================================

"""
Generate custom parameter definition templates for optimization.

NEW: Includes parabolic pattern threshold templates based on observed data ranges.
NEW: Added parabolic_9hour template (18 combinations for 9-hour runtime)
NEW: Added confluence_9hour template (18 hand-picked pattern weight combinations)
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_template(template_type: str, output_path: str):
    """Create a parameter template file."""
    
    templates = {
        'confluence_9hour': {
            'description': '9-hour pattern confluence weight optimization (18 explicit combinations × 30min)',
            # ✅ NEW: List combinations explicitly (not as arrays)
            'combinations_list': [
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 1.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 1.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 1.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.6, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.4, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.4, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.6, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.6, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.4, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.33, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.33, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.34, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.4, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.3, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.3, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.3, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.4, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.3, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.3, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.3, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.4, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.5, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.25, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.25, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.25, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.5, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.25, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.25, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.25, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.5, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.0, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.0, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.0, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 1.0, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.0},
                {'pattern.CONFLUENCE_WEIGHT_STEP_UP': 0.2, 'pattern.CONFLUENCE_WEIGHT_PARABOLIC': 0.2, 'pattern.CONFLUENCE_WEIGHT_VOLUME': 0.2, 'pattern.CONFLUENCE_WEIGHT_BREAKOUT': 0.2, 'pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE': 0.2},
            ]
        },
        'parabolic_9hour': {
            'description': '9-hour optimization run (18 combinations × 30min = 540min)',
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.67, -0.30, 0.0, 0.5, 1.0, 1.92],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 1.0, 1.2],
            }
        },
        'parabolic_threshold_quick': {
            'description': 'Quick test of parabolic angle thresholds (observed data range)',
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.42, -0.16, 0.0, 0.5, 1.0, 1.92],
                'PARABOLIC_MIN_ACCELERATION': [-0.00123, 0.0, 0.000472, 0.001],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 0.50, 1.0, 1.2],
            }
        },
        'parabolic_threshold_full': {
            'description': 'Comprehensive parabolic threshold testing',
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.67, -0.42, -0.16, 0.0, 0.5, 1.0, 1.5, 1.92, 2.5],
                'PARABOLIC_MAX_ANGLE': [3.0, 5.0, 10.0, 20.0],
                'PARABOLIC_MIN_ACCELERATION': [-0.00123, -0.0005, 0.0, 0.0002, 0.000472, 0.001, 0.002],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.13, 0.25, 0.50, 0.75, 1.0, 1.09, 1.2, 1.5, 2.0],
            }
        },
        'parabolic_angle_sweep': {
            'description': 'Fine-grained angle sweep (single parameter)',
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.67, -0.50, -0.42, -0.30, -0.16, 0.0, 0.2, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 1.92],
            }
        },
        'parabolic_with_step_up': {
            'description': 'Combined parabolic and step-up pattern optimization',
            'pattern': {
                'PARABOLIC_MIN_ANGLE': [-0.42, 0.0, 0.5, 1.0, 1.92],
                'PARABOLIC_MIN_ACCELERATION': [0.0, 0.001],
                'PARABOLIC_MIN_VOL_MULTIPLIER': [0.5, 1.0, 1.2],
                'MIN_STEP_UPS': [1, 2, 3],
                'MIN_ADVANCE_RETENTION': [25.0, 35.0, 45.0],
                'MAX_PULLBACK_PERCENT': [30.0, 40.0, 50.0],
            }
        },
        'pattern_focused': {
            'description': 'Traditional pattern-focused optimization',
            'pattern': {
                'MAX_PULLBACK_PERCENT': [30.0, 40.0, 50.0, 60.0],
                'MIN_ADVANCE_RETENTION': [25.0, 30.0, 35.0, 40.0, 45.0],
                'MIN_STEP_UPS': [1, 2, 3],
            }
        },
        'risk_focused': {
            'description': 'Risk management parameter optimization',
            'risk': {
                'STOP_LOSS_PERCENT_OF_ACCOUNT': [2.0, 3.0, 4.0, 5.0, 6.0],
                'MAX_HOLD_TIME_MINUTES': [15, 20, 30, 45, 60],
                'ATR_TRAILING_MULTIPLIER': [1.0, 1.5, 2.0, 2.5],
                'ATR_TRAILING_MIN_PROFIT_PCT': [0.5, 1.0, 1.5, 2.0],
            }
        },
        'screening_focused': {
            'description': 'Screening criteria optimization',
            'screening': {
                'MIN_GAP_PERCENT': [30.0, 40.0, 50.0, 60.0, 75.0, 100.0],
                'MIN_PRICE': [1.0, 2.0, 3.0, 5.0],
                'MAX_PRICE': [15.0, 20.0, 30.0, 50.0],
            }
        },
        'confluence_weights': {
            'description': 'Pattern confluence weight optimization',
            'pattern': {
                'CONFLUENCE_WEIGHT_STEP_UP': [0.0, 0.3, 0.5, 0.7, 1.0],
                'CONFLUENCE_WEIGHT_PARABOLIC': [0.0, 0.3, 0.5, 0.7, 1.0],
                'CONFLUENCE_WEIGHT_BREAKOUT': [0.0, 0.2, 0.4],
                'CONFLUENCE_WEIGHT_VOLUME': [0.0, 0.1, 0.2],
            }
        },
        'comprehensive': {
            'description': 'Comprehensive multi-parameter optimization',
            'pattern': {
                'MAX_PULLBACK_PERCENT': [30.0, 40.0, 50.0, 60.0],
                'PARABOLIC_MIN_ANGLE': [0.0, 0.5, 1.0],
                'MIN_ADVANCE_RETENTION': [30.0, 35.0, 40.0],
            },
            'risk': {
                'STOP_LOSS_PERCENT_OF_ACCOUNT': [3.0, 4.0, 5.0],
                'MAX_HOLD_TIME_MINUTES': [20, 30, 45],
            },
            'screening': {
                'MIN_GAP_PERCENT': [40.0, 50.0, 60.0],
            }
        },
        'single_param_test': {
            'description': 'Single parameter sweep (for sensitivity analysis)',
            'pattern': {
                'MAX_PULLBACK_PERCENT': [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0],
            }
        }
    }
    
    if template_type not in templates:
        print(f"ERROR: Unknown template type '{template_type}'")
        print(f"\nAvailable types:")
        for ttype, tdata in templates.items():
            desc = tdata.get('description', 'No description')
            print(f"  • {ttype}: {desc}")
        return 1
    
    template_data = templates[template_type]
    description = template_data.get('description', '')
    
    # Special handling for confluence_9hour (uses 'combinations' format)
    if template_type == 'confluence_9hour':
        # Export as explicit combination list
        template = template_data['combinations_list']
    else:
        # Remove description from template before saving
        template = {k: v for k, v in template_data.items() if k != 'description'}
    
    # Save to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Template created: {output.absolute()}")
    print(f"\nTemplate type: {template_type}")
    print(f"Description: {description}")
    
    # ✅ FIX: Handle both list and dict formats
    if isinstance(template, list):
        print(f"Format: Explicit combination list")
        print(f"Total combinations: {len(template)}")
    else:
        print(f"Parameter sections: {', '.join(template.keys())}")
    
    # Special display for confluence_9hour
    if template_type == 'confluence_9hour':
        print(f"\n📊 18 Hand-Picked Pattern Weight Combinations:")
        print("=" * 80)
        combinations = templates['confluence_9hour']['combinations_list']  # ✅ FIX: Use combinations_list
        for i, combo in enumerate(combinations, 1):
            step = combo['pattern.CONFLUENCE_WEIGHT_STEP_UP']
            para = combo['pattern.CONFLUENCE_WEIGHT_PARABOLIC']
            vol = combo['pattern.CONFLUENCE_WEIGHT_VOLUME']
            brk = combo['pattern.CONFLUENCE_WEIGHT_BREAKOUT']
            sr = combo['pattern.CONFLUENCE_WEIGHT_SUPPORT_RESISTANCE']
            
            # Determine combination description
            if step == 1.0:
                desc = "Pure Step-Up"
            elif para == 1.0:
                desc = "Pure Parabolic"
            elif vol == 1.0:
                desc = "Pure Volume"
            elif brk == 1.0:
                desc = "Pure Breakout"
            elif step == 0.5 and para == 0.5:
                desc = "Step-Up + Parabolic (50/50)"
            elif step == 0.5 and vol == 0.5:
                desc = "Step-Up + Volume (50/50)"
            elif para == 0.5 and vol == 0.5:
                desc = "Parabolic + Volume (50/50)"
            elif step == 0.6 and vol == 0.4:
                desc = "Step-Up + Volume (60/40)"
            elif step == 0.4 and vol == 0.6:
                desc = "Step-Up + Volume (40/60)"
            elif step == 0.6 and para == 0.4:
                desc = "Step-Up + Parabolic (60/40)"
            elif abs(step - 0.33) < 0.02 and abs(para - 0.33) < 0.02:
                desc = "All Three Equal (33/33/34)"
            elif all(w == 0.2 for w in [step, para, vol, brk, sr]):
                desc = "All Five Equal (20/20/20/20/20)"
            else:
                desc = f"Mixed (S={step:.0%}/P={para:.0%}/V={vol:.0%}/B={brk:.0%}/SR={sr:.0%})"
            
            print(f"  {i:2d}. {desc}")
            print(f"      Step={step:.2f}, Para={para:.2f}, Vol={vol:.2f}, Brk={brk:.2f}, SR={sr:.2f}")
        print("=" * 80)
        print(f"\n✓ Total: 18 combinations (NOT 5000+)")
        print(f"⏱️  Estimated runtime: 18 × 30 min = 540 minutes (9.0 hours)")
    else:
        # Calculate total combinations for other templates
        total = 1
        for section, params in template.items():
            section_total = 1
            for param_name, values in params.items():
                section_total *= len(values)
                total *= len(values)
            print(f"  {section}: {len(params)} parameters, {section_total} combinations")
        
        print(f"\nTotal combinations: {total:,}")
        
        # Time estimate
        if template_type in ['parabolic_9hour']:
            print(f"\n⏱️  Estimated runtime: {total} combinations × 30 min/year = {total * 30} minutes ({total * 30 / 60:.1f} hours)")
        elif total * 30 / 60 > 8:
            print(f"\n⏱️  Estimated runtime: {total} combinations × 30 min/year = {total * 30} minutes ({total * 30 / 60:.1f} hours)")
    
    if template_type != 'confluence_9hour':
        total_combos = 1
        for section, params in template.items():
            for param_name, values in params.items():
                total_combos *= len(values)
        
        if total_combos > 1000:
            print(f"\n⚠️  WARNING: {total_combos:,} combinations may take significant time to run")
            print("   Consider using a subset or running with --parallel flag")
    
    print("\nEdit this file to customize your parameter search space.")
    print(f"\nUsage:")
    print(f"  python scripts/pattern_optimization/optimize_parameters.py \\")
    print(f"    --preset custom --custom-params {output.name}")
    
    return 0


def list_templates():
    """List all available templates."""
    
    print("=" * 80)
    print("AVAILABLE PARAMETER TEMPLATES")
    print("=" * 80)
    print("\n⏱️  9-Hour Optimization Templates:")
    print("  • confluence_9hour - 18 hand-picked pattern weight combos (18 × 30min) ⭐ NEW")
    print("  • parabolic_9hour - Parabolic threshold sweep (18 combos × 30min)")
    
    print("\nParabolic Pattern Threshold Templates:")
    print("  • parabolic_threshold_quick - Quick parabolic threshold test (24 combinations)")
    print("  • parabolic_threshold_full - Comprehensive parabolic testing (504 combinations)")
    print("  • parabolic_angle_sweep - Fine-grained angle sweep (15 combinations)")
    print("  • parabolic_with_step_up - Combined parabolic + step-up (270 combinations)")
    
    print("\nTraditional Pattern Templates:")
    print("  • pattern_focused - Pattern recognition parameters (48 combinations)")
    print("  • risk_focused - Risk management parameters (80 combinations)")
    print("  • screening_focused - Screening criteria (96 combinations)")
    print("  • confluence_weights - Pattern weight optimization (400 combinations)")
    
    print("\nGeneral Templates:")
    print("  • comprehensive - Multi-parameter optimization (432 combinations)")
    print("  • single_param_test - Single parameter sensitivity (13 combinations)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate custom parameter templates for optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available templates
  python scripts/pattern_optimization/create_custom_params.py --list
  
  # Create 9-hour confluence weight optimization (18 combinations, not 5000!) ⭐ NEW
  python scripts/pattern_optimization/create_custom_params.py confluence_9hour
  
  # Create 9-hour parabolic threshold template
  python scripts/pattern_optimization/create_custom_params.py parabolic_9hour
  
  # Create template with custom output path
  python scripts/pattern_optimization/create_custom_params.py confluence_9hour \\
    --output my_confluence_test.json
        """
    )
    parser.add_argument(
        'template_type',
        nargs='?',
        help="Type of template to generate (use --list to see options)"
    )
    parser.add_argument(
        '--output',
        default='custom_params.json',
        help="Output filename (default: custom_params.json)"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available templates"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_templates()
        return 0
    
    if not args.template_type:
        print("ERROR: template_type required (use --list to see options)")
        parser.print_help()
        return 1
    
    return create_template(args.template_type, args.output)


if __name__ == "__main__":
    exit(main())
