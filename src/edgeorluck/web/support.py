"""Supporting edge or luck, and what supporters unlock: Flip's wardrobe and the Flippy games.

Support is one click away: Buy Me a Coffee (EDGEORLUCK_BMC_URL) for a coffee now and then, Patreon
(EDGEORLUCK_PATREON_URL) for a little every month. Each platform shows its supporters a thank-you message, and that
message carries a thank-you link: /thanks/<token>, signed with EDGEORLUCK_SECRET. Opening it unlocks that tier in
the browser, one click and nothing to type. Make the links with

    python -m edgeorluck.web.support link 3          # tier 3's link (run where EDGEORLUCK_SECRET is set)

and paste each into the matching thank-you message (Buy Me a Coffee: tier 1; Patreon: one per membership tier).
Nothing about supporters is stored here. The extras are for fun: the judge is the same for everyone.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
import sys
from typing import Dict, List, Optional

from .guides import _shell

# (name, how to get it): a coffee once, then Patreon memberships. The more someone gives, the more Flip gets.
TIERS = [("Coffee", "a coffee, once"), ("Fan", "3 € a month"), ("Patron", "5 € a month"), ("Champion", "10 € a month")]

# Flip's wardrobe: id, what it is, which part of Flip it goes on, and the tier that unlocks it (1 to 4).
ITEMS = [
    {"id": "shades", "name": "Sunglasses", "slot": "face", "tier": 1},
    {"id": "party", "name": "Party hat", "slot": "hat", "tier": 1},
    {"id": "wink", "name": "A wink", "slot": "emote", "tier": 1},
    {"id": "wave", "name": "Waving", "slot": "move", "tier": 1},
    {"id": "cap", "name": "Cap", "slot": "hat", "tier": 2},
    {"id": "bowtie", "name": "Bow tie", "slot": "neck", "tier": 2},
    {"id": "stars", "name": "Star eyes", "slot": "emote", "tier": 2},
    {"id": "dance", "name": "A little dance", "slot": "move", "tier": 2},
    {"id": "tophat", "name": "Top hat", "slot": "hat", "tier": 3},
    {"id": "scarf", "name": "Scarf", "slot": "neck", "tier": 3},
    {"id": "hearts", "name": "Heart eyes", "slot": "emote", "tier": 3},
    {"id": "spin", "name": "A spin", "slot": "move", "tier": 3},
    {"id": "gold", "name": "Golden coin", "slot": "skin", "tier": 3},
    {"id": "crown", "name": "Crown", "slot": "hat", "tier": 4},
    {"id": "wizard", "name": "Wizard hat", "slot": "hat", "tier": 4},
    {"id": "cape", "name": "Cape", "slot": "back", "tier": 4},
    {"id": "juggle", "name": "Juggling coins", "slot": "move", "tier": 4},
    {"id": "rainbow", "name": "Rainbow rim", "slot": "skin", "tier": 4},
]
SLOTS = [("hat", "On his head"), ("face", "On his face"), ("neck", "Round his neck"), ("back", "On his back"),
         ("skin", "His coin"), ("emote", "When you tap him: a face"), ("move", "When you tap him: a move")]
# The Flippy games: small games with Flip, more of them the more someone supports (/games, static/assets/games.js).
GAMES = [
    {"id": "catch", "name": "Catch the coins", "tier": 1,
     "how": "Move Flip with the mouse, your finger or the arrow keys. Gold coins are points; three red candles and it's over."},
    {"id": "hop", "name": "Bull run", "tier": 2,
     "how": "Tap, click or press space to hop over the red candles. The market speeds up as you go."},
    {"id": "guess", "name": "Edge or luck?", "tier": 3,
     "how": "Two trading records: one has a real edge, the other is plain luck. Pick the edge, ten rounds."},
]


def _signature(body: str) -> Optional[str]:
    """The signature of a thank-you token's body, or None without EDGEORLUCK_SECRET (then no link works)."""
    secret = os.environ.get("EDGEORLUCK_SECRET")
    return hmac.new(secret.encode(), b"thanks:" + body.encode(), hashlib.sha256).hexdigest()[:20] if secret else None


def thanks_token(tier: int) -> str:
    """A thank-you link's token for a tier: tier, a random part (each link different), and a signature."""
    if not 1 <= tier <= len(TIERS):
        raise ValueError(f"tier is 1 to {len(TIERS)}")
    body = f"{tier}.{secrets.token_urlsafe(6)}"
    sig = _signature(body)
    if sig is None:
        raise RuntimeError("Set EDGEORLUCK_SECRET to the site's own: the links are signed with it.")
    return f"{body}.{sig}"


def tier_of_token(token: str) -> int:
    """The tier a thank-you link unlocks, or 0 when it isn't one this site signed."""
    body, _, sig = str(token).rpartition(".")
    tier, good = body.split(".")[0], _signature(body)
    ok = bool(body) and good is not None and hmac.compare_digest(sig, good) and tier.isdigit()
    return int(tier) if ok and 1 <= int(tier) <= len(TIERS) else 0


def _best(tier: int, tricks: bool = False) -> Dict[str, str]:
    """The outfit that shows a tier off: per part of Flip, the first item of the highest tier reached (with tricks:
    his face and move too)."""
    outfit: Dict[str, str] = {}
    best: Dict[str, int] = {}
    for item in ITEMS:
        if item["tier"] <= tier and (tricks or item["slot"] not in ("emote", "move"))                 and item["tier"] > best.get(item["slot"], 0):
            outfit[item["slot"]], best[item["slot"]] = item["id"], item["tier"]
    return outfit


def _buttons() -> str:
    e = html.escape
    bmc, patreon = os.environ.get("EDGEORLUCK_BMC_URL"), os.environ.get("EDGEORLUCK_PATREON_URL")
    links = "".join(f'<a class="btn{cls}" href="{e(url)}" rel="noopener">{label}</a>' for url, label, cls in
                    ((bmc, "☕ Buy me a coffee", ""), (patreon, "Join on Patreon", " quiet")) if url)
    return f'<p class="give">{links}</p>' if links else '<p class="give"><span class="soon">Supporting opens soon.</span></p>'


def _tier_card(n: int, name: str, how: str) -> str:
    e = html.escape
    things = [item["name"] for item in ITEMS if item["tier"] == n] + [f"Game: {g['name']}" for g in GAMES if g["tier"] == n]
    return (f'<li class="tier"><div class="pet tier-pet" data-outfit="{e(json.dumps(_best(n)))}"></div><h3>{e(name)}</h3>'
            f'<p class="at">{e(how)}</p><ul>{"".join(f"<li>{e(t)}</li>" for t in things)}</ul></li>')


def page() -> str:
    tiers = "".join(_tier_card(n, name, how) for n, (name, how) in enumerate(TIERS, start=1))
    body = f"""<section class="intro wrap" aria-labelledby="title">
  <div>
    <h1 id="title">Support edge or luck</h1>
    <p class="lede">If edge or luck kept you from trading a lucky strategy, you can keep it going. As a thank-you, Flip gets new things to wear, new tricks and new games, and the more you give, the more he gets.</p>
    {_buttons()}
    <p class="for-ai">After you support, open the thank-you link you get: that's all, it unlocks everything for your tier here. The extras are for fun; the judge is the same for everyone.</p>
  </div>
  <div class="intro-pet" aria-hidden="true"><div class="pet big-pet" id="wardrobePet"></div></div>
</section>
<section class="wrap support" aria-labelledby="tiers-title">
  <h2 id="tiers-title">What you get</h2>
  <ul class="tiers">{tiers}</ul>
  <section class="wardrobe" id="wardrobe" aria-labelledby="wardrobe-title">
    <h2 id="wardrobe-title">Flip's wardrobe</h2>
    <p id="tierNow">Nothing unlocked here yet.</p>
    <div class="slots" id="slots"></div>
    <p class="hint">Flip wears it on every page. Tap him anywhere to see his face and move. <a href="/games">The Flippy games →</a></p>
  </section>
</section>
<script>window.EOL_WARDROBE = {json.dumps({"items": ITEMS, "slots": SLOTS, "tiers": [n for n, _ in TIERS]})};</script>
<script src="/assets/support.js" defer></script>"""
    return _shell("Support · edge or luck", "Support edge or luck, and give Flip new things to wear and new games.",
                  body, current="/support", css=("support.css",))


def thanks_page(tier: int) -> str:
    """What a thank-you link opens: the tier, unlocked in this browser (never lowered), then the wardrobe."""
    e = html.escape
    if not tier:
        body = ('<article class="wrap guide"><header class="guide-head"><div><h1>That link doesn\'t work</h1><p class="lede">'
                'It may be cut short, or an old one. Open it again from the thank-you message, or write to edge or luck on '
                'Buy Me a Coffee or Patreon.</p></div></header></article>')
        return _shell("Thank you · edge or luck", "A thank-you link.", body, current="/support")
    name = TIERS[tier - 1][0]
    body = f"""<section class="intro wrap" aria-labelledby="title">
  <div>
    <h1 id="title">Thank you!</h1>
    <p class="lede">Unlocked here: <b>{e(name)}</b>. Flip is already wearing some of it: change it in his wardrobe, and play the games that opened for you.</p>
    <p class="give"><a class="btn" href="/support#wardrobe">Dress Flip up</a> <a class="btn quiet" href="/games">Play the Flippy games</a></p>
  </div>
  <div class="intro-pet" aria-hidden="true"><div class="pet big-pet" data-mood="happy"></div></div>
</section>
<script>try {{
  if (+(localStorage.getItem("eol-tier") || 0) < {tier}) {{ localStorage.setItem("eol-tier", "{tier}"); localStorage.setItem("eol-flip", {json.dumps(json.dumps(_best(tier, tricks=True)))}); }}
}} catch (e) {{}}</script>"""
    return _shell("Thank you · edge or luck", "Thank you for supporting edge or luck.", body, current="/support",
                  css=("support.css",))


def games_page() -> str:
    body = f"""<section class="wrap games" aria-labelledby="title">
  <h1 id="title">The Flippy games</h1>
  <p class="lede">Small games starring Flip, a thank-you to supporters: the more you support, the more games. <a href="/play">Flip Flip</a> is free for everyone.</p>
  <div class="picker" id="picker"></div>
  <p class="say" id="say" aria-live="polite"></p>
  <section class="stage" id="stage" aria-label="The game">
    <canvas id="cv" aria-hidden="true"></canvas>
    <div class="pet stage-flip" id="stageFlip" aria-hidden="true"></div>
    <p class="hud" id="hud"></p>
    <div class="picks" id="picks" hidden><button class="btn" type="button" data-pick="0">A has the edge</button><button class="btn" type="button" data-pick="1">B has the edge</button><button class="btn next" type="button" id="next">Next round</button></div>
    <div class="overlay" id="overlay"><h2 id="ovTitle"></h2><p id="ovText"></p><div class="ov-actions" id="ovActions"></div></div>
  </section>
</section>
<script>window.EOL_GAMES = {json.dumps({"games": GAMES, "tiers": [n for n, _ in TIERS]})};</script>
<script src="/assets/games.js" defer></script>"""
    return _shell("The Flippy games · edge or luck", "Small games starring Flip, the referee coin: a thank-you to "
                  "supporters of edge or luck.", body, current="/games", css=("games.css",))


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "link" or not args[1].isdigit():
        print("usage: python -m edgeorluck.web.support link <tier 1-4>", file=sys.stderr)
        return 2
    try:
        token = thanks_token(int(args[1]))
    except (ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{os.environ.get('EDGEORLUCK_URL', '').rstrip('/')}/thanks/{token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["TIERS", "ITEMS", "SLOTS", "GAMES", "thanks_token", "tier_of_token", "page", "thanks_page", "games_page"]
