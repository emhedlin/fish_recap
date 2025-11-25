"""Script to verify Hugging Face authentication and SAM 3 model access.

This script checks:
1. Hugging Face authentication status
2. SAM 3 package installation
3. Access to the SAM 3 model repository
4. Ability to load the SAM 3 model

Run this script after setting up Hugging Face authentication to verify everything works.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face."""
    print("=" * 70)
    print("Checking Hugging Face Authentication...")
    print("=" * 70)
    
    try:
        from huggingface_hub import whoami, HfApi
        
        try:
            user_info = whoami()
            username = user_info.get("name", "Unknown")
            print(f"✓ Authenticated as: {username}")
            print(f"  Full name: {user_info.get('fullname', 'N/A')}")
            print(f"  Email: {user_info.get('email', 'N/A')}")
            return True
        except Exception as e:
            print("✗ Not authenticated")
            print(f"  Error: {e}")
            print("\n  To authenticate, run:")
            print("    huggingface-cli login")
            print("  Or set HUGGING_FACE_HUB_TOKEN environment variable")
            return False
            
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("  Install with: uv pip install huggingface_hub")
        return False


def check_sam3_installation():
    """Check if SAM 3 package is installed."""
    print("\n" + "=" * 70)
    print("Checking SAM 3 Installation...")
    print("=" * 70)
    
    try:
        import sam3
        sam3_path = os.path.dirname(sam3.__file__)
        print(f"✓ SAM 3 installed at: {sam3_path}")
        
        # Check if BPE file exists
        sam3_root = os.path.join(sam3_path, "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(bpe_path):
            print(f"✓ BPE vocabulary file found")
        else:
            print(f"⚠ BPE vocabulary file not found at: {bpe_path}")
            print("  This may cause issues. Model will use default.")
        
        return True
        
    except ImportError as e:
        print("✗ SAM 3 not installed")
        print(f"  Error: {e}")
        print("\n  To install SAM 3:")
        print("    1. Clone repository: git clone https://github.com/facebookresearch/sam3.git")
        print("    2. Install: cd sam3 && uv pip install -e .")
        return False


def check_sam3_model_access():
    """Check if user has access to SAM 3 model repository."""
    print("\n" + "=" * 70)
    print("Checking SAM 3 Model Access...")
    print("=" * 70)
    
    try:
        from huggingface_hub import HfApi, model_info
        
        model_id = "facebook/sam3"
        
        try:
            info = model_info(model_id)
            print(f"✓ Model repository accessible: {model_id}")
            
            # Check if model is gated
            if hasattr(info, "gated") and info.gated:
                print("  Model is gated (requires access approval)")
                
                # Try to check if user has access by attempting to list files
                try:
                    api = HfApi()
                    files = api.list_repo_files(model_id, repo_type="model")
                    print(f"✓ Access granted! Found {len(files)} files in repository")
                    return True
                except Exception as e:
                    error_str = str(e).lower()
                    if "403" in error_str or "gated" in error_str or "authorized" in error_str:
                        print("✗ Access not granted yet")
                        print("\n  To request access:")
                        print(f"    1. Visit: https://huggingface.co/{model_id}")
                        print("    2. Click 'Request access' button")
                        print("    3. Wait for approval (usually instant)")
                        return False
                    else:
                        print(f"⚠ Could not verify file access: {e}")
                        return None
            else:
                print("  Model is public (no access required)")
                return True
                
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "gated" in error_str:
                print(f"✗ Access denied to {model_id}")
                print("\n  To request access:")
                print(f"    1. Visit: https://huggingface.co/{model_id}")
                print("    2. Click 'Request access' button")
                print("    3. Wait for approval (usually instant)")
                return False
            else:
                print(f"✗ Error checking model access: {e}")
                return False
                
    except ImportError:
        print("✗ huggingface_hub not installed")
        return False


def check_sam3_model_load():
    """Check if SAM 3 model can be loaded."""
    print("\n" + "=" * 70)
    print("Checking SAM 3 Model Loading...")
    print("=" * 70)
    
    try:
        import torch
        from sam3 import build_sam3_image_model
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        if device == "cpu":
            print("  ⚠ GPU not available, using CPU (will be slower)")
        
        # Try to load model
        print("  Attempting to load model...")
        try:
            model = build_sam3_image_model()
            model = model.to(device)
            model.eval()
            print("✓ Model loaded successfully!")
            
            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {param_count / 1e9:.2f}B")
            print(f"  Device: {next(model.parameters()).device}")
            
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "gated" in error_str or "authorized" in error_str:
                print("✗ Model access denied")
                print("  This usually means:")
                print("    1. You haven't requested access to the model")
                print("    2. Your access request hasn't been approved yet")
                print("    3. You're not authenticated properly")
                print("\n  See troubleshooting steps above.")
                return False
            else:
                print(f"✗ Failed to load model: {e}")
                print("\n  This could be due to:")
                print("    - Network connectivity issues")
                print("    - Insufficient disk space")
                print("    - Corrupted model cache")
                return False
                
    except ImportError as e:
        print(f"✗ SAM 3 not available: {e}")
        return False


def main():
    """Run all checks and provide summary."""
    print("\n" + "=" * 70)
    print("Fish Re-Identification Project - Hugging Face & SAM 3 Access Check")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Run checks
    results["auth"] = check_huggingface_auth()
    results["installation"] = check_sam3_installation()
    results["access"] = check_sam3_model_access()
    results["load"] = check_sam3_model_load()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = all(v for v in results.values() if v is not None)
    any_failed = any(v is False for v in results.values())
    
    if all_passed:
        print("✓ All checks passed! You're ready to use SAM 3.")
        return 0
    elif any_failed:
        print("✗ Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  1. Run: huggingface-cli login")
        print("  2. Request access at: https://huggingface.co/facebook/sam3")
        print("  3. Wait for access approval")
        print("  4. Run this script again to verify")
        return 1
    else:
        print("⚠ Some checks had warnings. Review the output above.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

