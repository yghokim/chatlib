from os import path, getcwd, getenv

from dotenv import load_dotenv


def get_env_variable(key: str) -> str:
    load_dotenv(path.join(getcwd(), ".env"))
    return getenv(key)
