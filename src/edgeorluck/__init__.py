"""edge or luck: is a trading record a real edge, or luck?

    import edgeorluck
    result = edgeorluck.judge("trades.csv", variants_tried=40)
    print(result["verdict"]["headline"])
    print(result.to_markdown())

An idea with no trades yet? Test it honestly first, then the same judge decides:

    px = edgeorluck.prices("BTCUSDT", "1h", start="2024-01-01")
    print(edgeorluck.backtest(px, {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}).to_markdown())
    print(edgeorluck.tune(px, {"buy": "rsi(14) < {low}", "sell": "rsi(14) > 70"},
                          {"low": [20, 25, 30]}).to_markdown())

Pure: reads its input, writes nothing, and the same input always gives the
same result (prices() fetches public market data when asked). Not financial
advice: a statistical check of past trades.
"""
from .judge import CHANGELOG, ENGINE_VERSION, SCHEMA_VERSION
from .market import PriceError, load_prices, prices
from .result import Result, judge
from .rulebook import RULE_FUNCTIONS, Rule, RuleError
from .schema import schema
from .simulate import Backtest, Tuned, backtest, tune

__version__ = ENGINE_VERSION
__all__ = ["judge", "backtest", "tune", "prices", "load_prices", "schema", "Result", "Backtest", "Tuned", "Rule",
           "RuleError", "PriceError", "RULE_FUNCTIONS", "ENGINE_VERSION", "SCHEMA_VERSION", "CHANGELOG", "__version__"]
