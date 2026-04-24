"""Allow running as `python -m lilbee`."""

if __name__ == "__main__":
    # Must run before any import that triggers multiprocessing so a spawned
    # subprocess (e.g. resource_tracker) in a PyInstaller frozen bundle
    # doesn't re-enter the typer CLI and reject -B/-s/-E as unknown options.
    import multiprocessing

    multiprocessing.freeze_support()

    from lilbee.launcher import main

    main()
