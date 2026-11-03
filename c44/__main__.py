from .cli import main

if __name__ == "__main__":          # guard: spawn-based worker pools re-import this module and must not re-run the CLI
    main()
