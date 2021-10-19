from pathlib import Path

from dotenv import load_dotenv

# Assumes config.env is at the same lvl as package root
load_dotenv(Path(__path__[0]).parent / 'config.env')
