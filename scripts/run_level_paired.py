try:
    from .run_level1_paired import main
except ImportError:
    from run_level1_paired import main


if __name__ == "__main__":
    main()
