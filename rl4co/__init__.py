try:
    from importlib.metadata import version as get_version
    __version__ = get_version(__package__)
except Exception:
    __version__ = "0.6.0"
