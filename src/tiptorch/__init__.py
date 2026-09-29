from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tiptorch")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
