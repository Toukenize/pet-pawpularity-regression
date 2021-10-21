from pathlib import Path

from dotenv import load_dotenv

# Assumes config.env is at the same lvl as package root

_root = Path(__path__[0]).parent

for env_file in ['config.env', 'secrets.env']:

    if (_root / env_file).exists():
        load_dotenv(_root / env_file)
