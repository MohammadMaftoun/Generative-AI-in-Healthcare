# ============================================================================
# FILE: main.py
# CLI interface for synthetic medical imaging system
# ============================================================================

import argparse
import sys
from datetime import datetime
from config import Config
from prompt_generator import MedicalPromptGenerator
from image_generator import MedicalImageGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical images using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"\"\"
Examples:
  python main.py --modality xray --region chest --count 3
  python main.py --modality mri --region brain --detail high --resolution 768
  python main.py --modality ct --region abdomen --angle axial --llm claude
        \"\"\"
    )
    
    # Required arguments
    parser.add_argument(
        "--modality",
        type=str,
        default="xray",
        choices=list(Config.MODALITIES.keys()),
        help="Imaging modality"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="chest",
        choices=Config.BODY_REGIONS,
        help="Body region to image"
    )
    
    # Optional arguments
    parser.add_argument(
        "--detail",
        type=str,
        default="medium",
        choices=list(Config.DETAIL_LEVELS.keys()),
        help="Level of detail"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=Config.SUPPORTED_RESOLUTIONS,
        help="Image resolution (width and height)"
    )
    parser.add_argument(
        "--angle",
        type=str,
        help="Imaging angle/view (auto-selected if not provided)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="claude",
        choices=Config.LLM_PROVIDERS,
        help="LLM provider for prompt generation"
    )
    parser.add_argument(
        "--safety-mode",
        type=str,
        default="strict",
        choices=["strict", "moderate", "permissive"],
        help="Safety filter mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of inference steps (overrides detail level default)"
    )
    
    return parser.parse_args()

def print_header():
    """Print application header."""
    print("=" * 70)
    print("  SYNTHETIC MEDICAL IMAGING SYSTEM")
    print("  Powered by Stable Diffusion + LLMs")
    print("=" * 70)
    print("\\n⚠️  WARNING: All outputs are synthetic and for research use only")
    print("   Not for diagnostic or clinical purposes\\n")

def main():
    """Main application entry point."""
    args = parse_args()
    
    print_header()
    
    # Setup directories
    Config.setup_directories()
    
    # Initialize components
    try:
        print("[1/4] Initializing prompt generator...")
        prompt_gen = MedicalPromptGenerator(
            llm_provider=args.llm,
            safety_mode=args.safety_mode
        )
        print("✓ Prompt generator ready\\n")
        
        print("[2/4] Generating medical prompt with LLM...")
        prompt = prompt_gen.generate_prompt(
            modality=args.modality,
            body_region=args.region,
            detail_level=args.detail,
            angle=args.angle
        )
        print(f"✓ Prompt generated\\n")
        print("Generated Prompt:")
        print("-" * 70)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 70 + "\\n")
        
        print("[3/4] Initializing image generator...")
        image_gen = MedicalImageGenerator(
            precision=Config.STABLE_DIFFUSION_PRECISION
        )
        
        print("\\n[4/4] Generating images...")
        
        # Determine number of steps
        num_steps = args.steps or Config.DETAIL_LEVELS[args.detail]["num_inference_steps"]
        
        # Generate images
        images = image_gen.generate(
            prompt=prompt,
            num_images=args.count,
            resolution=args.resolution,
            num_inference_steps=num_steps,
            guidance_scale=Config.DEFAULT_GUIDANCE_SCALE,
            seed=args.seed
        )
        
        # Save images
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "modality": args.modality,
            "body_region": args.region,
            "detail_level": args.detail,
            "resolution": args.resolution,
            "num_images": args.count,
            "prompt": prompt,
            "llm_provider": args.llm,
            "seed": args.seed,
            "inference_steps": num_steps,
            "guidance_scale": Config.DEFAULT_GUIDANCE_SCALE,
            "model": Config.STABLE_DIFFUSION_MODEL,
            "safety_notice": "SYNTHETIC - NOT FOR DIAGNOSTIC USE"
        }
        
        output_dir = image_gen.save_images(
            images=images,
            modality=args.modality,
            region=args.region,
            metadata=metadata
        )
        
        print("\\n" + "=" * 70)
        print("✓ Generation complete!")
        print(f"  Output: {output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
