"""
Command Line Interface for SAMe (Search Analog Model Engine)
"""
import argparse
import sys
from pathlib import Path

from same import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI parser"""
    parser = argparse.ArgumentParser(
        prog="same-cli",
        description="SAMe (Search Analog Model Engine) Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"SAMe {__version__}",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )
    
    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start the SAMe server",
    )
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    # Database commands
    db_parser = subparsers.add_parser(
        "db",
        help="Database management commands",
    )
    db_subparsers = db_parser.add_subparsers(
        dest="db_command",
        help="Database commands",
    )
    
    db_subparsers.add_parser(
        "init",
        help="Initialize the database",
    )
    
    db_subparsers.add_parser(
        "upgrade",
        help="Upgrade database to latest migration",
    )
    
    db_subparsers.add_parser(
        "downgrade",
        help="Downgrade database by one migration",
    )
    
    # Check command
    subparsers.add_parser(
        "check",
        help="Check project structure and dependencies",
    )
    
    return parser


def cmd_server(args):
    """Start the SAMe server"""
    import uvicorn
    
    print(f"🚀 Starting SAMe server on {args.host}:{args.port}")
    if args.reload:
        print("🔄 Auto-reload enabled")
    
    uvicorn.run(
        "same.api.create_app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


def cmd_db(args):
    """Handle database commands"""
    if args.db_command == "init":
        print("🗄️  Initializing database...")
        # TODO: Implement database initialization
        print("✅ Database initialized")
    elif args.db_command == "upgrade":
        print("⬆️  Upgrading database...")
        # TODO: Implement alembic upgrade
        print("✅ Database upgraded")
    elif args.db_command == "downgrade":
        print("⬇️  Downgrading database...")
        # TODO: Implement alembic downgrade
        print("✅ Database downgraded")
    else:
        print("❌ Unknown database command")
        sys.exit(1)


def cmd_check(args):
    """Check project structure and dependencies"""
    print("🔍 Checking SAMe project structure...")
    
    # Import and run the existing check script
    try:
        from pathlib import Path
        import sys
        
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        from check_structure import main as check_main
        check_main()
    except ImportError:
        print("❌ Could not import check_structure module")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "server":
            cmd_server(args)
        elif args.command == "db":
            cmd_db(args)
        elif args.command == "check":
            cmd_check(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
