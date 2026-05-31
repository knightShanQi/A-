from a_share_predictor.database_source import load_env_file


if __name__ == "__main__":
    load_env_file()
    from a_share_predictor.api import run

    run()
