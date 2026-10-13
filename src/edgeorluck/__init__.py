"""edge or luck: is a trading record a real edge, or luck?

    import edgeorluck
    result = edgeorluck.judge("trades.csv", variants_tried=40)
    print(result["verdict"]["headline"])
    print(result.to_markdown())

Pure: reads its input, writes nothing, and the same input always gives the
same result. Not financial advice: a statistical check of past trades.
"""
from .judge import CHANGELOG, ENGINE_VERSION, SCHEMA_VERSION
from .result import Result, judge
from .schema import schema

__version__ = ENGINE_VERSION
__all__ = ["judge", "schema", "Result", "ENGINE_VERSION", "SCHEMA_VERSION", "CHANGELOG", "__version__"]
