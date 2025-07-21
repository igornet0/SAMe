#!/usr/bin/env python3
"""
SAMe Project Test Runner
========================

This script provides stable test execution for the SAMe project, avoiding segmentation faults
that can occur when running all 274 tests simultaneously.

Usage:
    python run_tests.py [options]

Options:
    --core          Run core functionality tests (122 tests)
    --api           Run API tests (key functionality)
    --collect       Test collection only (verify imports)
    --groups        Run tests in stable groups
    --all           Attempt to run all tests (may cause segfault)
    --help          Show this help message

Examples:
    python run_tests.py --core      # Run 122 core functionality tests
    python run_tests.py --api       # Run key API tests
    python run_tests.py --groups    # Run all tests in stable groups
    python run_tests.py --collect   # Verify all imports work
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=Path.cwd())
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def run_core_tests():
    """Run the 122 core functionality tests"""
    cmd = [
        "poetry", "run", "pytest",
        "tests/test_data_manager.py",
        "tests/test_export.py", 
        "tests/test_parameter_extraction.py",
        "tests/test_search_engine.py",
        "tests/test_text_processing.py",
        "tests/test_utils.py",
        "--tb=short"
    ]
    return run_command(cmd, "Core Functionality Tests (122 tests)")


def run_api_tests():
    """Run key API tests"""
    cmd = [
        "poetry", "run", "pytest",
        "tests/test_api_endpoints.py::TestMainEndpoints::test_root_endpoint",
        "tests/test_file_upload_api.py::TestFileUploadValidation::test_upload_unsupported_file_format",
        "tests/test_api_integration.py::TestFullAPIWorkflow::test_complete_search_workflow",
        "tests/test_api_middleware_security.py::TestCORSMiddleware::test_cors_headers_present",
        "-v"
    ]
    return run_command(cmd, "Key API Tests (4 tests)")


def test_collection():
    """Test that all tests can be collected (verifies imports)"""
    cmd = ["poetry", "run", "pytest", "--collect-only", "-q"]
    return run_command(cmd, "Test Collection (Import Verification)")


def run_tests_in_groups():
    """Run tests in stable groups to avoid segfaults"""
    groups = [
        {
            "name": "Core Functionality Group 1",
            "tests": ["tests/test_data_manager.py", "tests/test_export.py", "tests/test_parameter_extraction.py"]
        },
        {
            "name": "Core Functionality Group 2", 
            "tests": ["tests/test_search_engine.py", "tests/test_text_processing.py", "tests/test_utils.py"]
        },
        {
            "name": "API Tests Group",
            "tests": ["tests/test_api_endpoints.py", "tests/test_api_integration.py", "tests/test_api_middleware_security.py", "tests/test_file_upload_api.py"]
        },
        {
            "name": "Database and Import Tests",
            "tests": ["tests/test_database.py", "tests/test_excel_import.py", "tests/test_settings.py"]
        },
        {
            "name": "Notebook Tests",
            "tests": ["tests/test_notebook_imports.py", "tests/test_notebook_lemmatizer_cell.py", "tests/test_notebook_modules.py", "tests/test_notebook_normalizer_cell.py", "tests/test_simple_notebook.py"]
        },
        {
            "name": "Advanced Tests",
            "tests": ["tests/test_advanced_model_manager.py", "tests/test_model_manager_integration.py", "tests/test_websocket_endpoints.py"]
        }
    ]
    
    all_passed = True
    results = []
    
    for group in groups:
        cmd = ["poetry", "run", "pytest"] + group["tests"] + ["--tb=short"]
        success = run_command(cmd, group["name"])
        results.append((group["name"], success))
        if not success:
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 GROUP TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall Result: {'✅ ALL GROUPS PASSED' if all_passed else '❌ SOME GROUPS FAILED'}")
    return all_passed


def run_all_tests():
    """Attempt to run all tests (may cause segfault)"""
    print("⚠️  WARNING: Running all tests simultaneously may cause segmentation fault")
    print("   Consider using --groups option instead for stable execution")
    
    response = input("Continue anyway? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return False
    
    cmd = ["poetry", "run", "pytest", "--tb=short"]
    return run_command(cmd, "All Tests (274 tests - may cause segfault)")


def main():
    parser = argparse.ArgumentParser(
        description="SAMe Project Test Runner - Stable test execution avoiding segfaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--core", action="store_true", help="Run core functionality tests (122 tests)")
    parser.add_argument("--api", action="store_true", help="Run key API tests")
    parser.add_argument("--collect", action="store_true", help="Test collection only (verify imports)")
    parser.add_argument("--groups", action="store_true", help="Run tests in stable groups")
    parser.add_argument("--all", action="store_true", help="Attempt to run all tests (may cause segfault)")
    
    args = parser.parse_args()
    
    if not any([args.core, args.api, args.collect, args.groups, args.all]):
        print("🚀 SAMe Project Test Runner")
        print("No option specified. Use --help for usage information.")
        print("\nQuick options:")
        print("  --core     Run 122 core functionality tests (recommended)")
        print("  --api      Run key API tests")
        print("  --groups   Run all tests in stable groups")
        print("  --collect  Verify imports work")
        return
    
    success = True
    
    if args.collect:
        success &= test_collection()
    
    if args.core:
        success &= run_core_tests()
    
    if args.api:
        success &= run_api_tests()
    
    if args.groups:
        success &= run_tests_in_groups()
    
    if args.all:
        success &= run_all_tests()
    
    print(f"\n{'='*60}")
    print(f"🏁 FINAL RESULT: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
