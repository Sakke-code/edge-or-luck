/* edge or luck: the Flippy games, small games starring Flip: the first free, the rest a thank-you to supporters.
   support.py lists them and the tier each one opens at; a supporter's thank-you link sets the tier in this browser.
   Best scores stay here too. */
(function () {
  "use strict";
  var DATA = window.EOL_GAMES, $ = function (id) { return document.getElementById(id); };
  var W = 640, H = 360, GROUND = H - 34;
  var stage = $("stage"), cv = $("cv"), ctx = cv.getContext("2d"), host = $("stageFlip");
  var flip = EOL.makeFlip(host, null), FW = 80, C = {}, current = null, game = null, playing = false, last = 0, moodTimer = 0;

  function tier() { try { return +localStorage.getItem("eol-tier") || 0; } catch (e) { return 0; } }
  function best(id, score) {          // the best score here, after counting this one
    var b = 0;
    try {
      b = +localStorage.getItem("eol-best-" + id) || 0;
      if (score > b) { b = score; localStorage.setItem("eol-best-" + id, String(b)); }
    } catch (e) { b = Math.max(b, score || 0); }
    return b;
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function rnd(a, b) { return a + Math.random() * (b - a); }
  function normal() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
  function sum(t) { return t.reduce(function (a, b) { return a + b; }, 0); }
  function colors() {
    ["ink", "ink-2", "ink-3", "paper", "rule", "red", "green", "amber"].forEach(function (k) { C[k] = EOL.cssVar("--" + k); });
    C.hand = EOL.cssVar("--hand") || "cursive";
  }

  /* ---------- the stage ---------- */
  function fit() {            // phones get a squarer stage, so the words stay readable
    var w = stage.clientWidth, r = window.devicePixelRatio || 1;
    W = w < 560 ? 480 : 640;
    stage.style.aspectRatio = W + " / " + H;
    cv.width = Math.round(w * r); cv.height = Math.round(w * r * H / W);
    ctx.setTransform(cv.width / W, 0, 0, cv.width / W, 0, 0);
    size(FW);
    draw();
  }
  function size(w) { FW = w; host.style.width = (w / W * 100) + "%"; }
  function place(x, feet) {   // Flip's feet at (x, feet), in game units
    host.style.left = (x / W * 100) + "%";
    host.style.top = ((feet - FW * 90 / 84 * 0.95) / H * 100) + "%";
  }
  function feel(mood, ms) {  // a mood, for a moment when ms is given
    clearTimeout(moodTimer);
    flip.mood(mood);
    if (ms) moodTimer = setTimeout(function () { flip.mood("idle"); }, ms);
  }
  function say(t) { $("say").textContent = t || ""; }
  function hud(t) { $("hud").textContent = t ? t + " · best " + best(current.id, 0) : ""; }
  function draw() { ctx.clearRect(0, 0, W, H); if (game) game.draw(); }
  function text(t, x, y, px, color, align) {
    ctx.font = "700 " + px + "px " + C.hand;
    ctx.fillStyle = color || C.ink;
    ctx.textAlign = align || "left";
    ctx.fillText(t, x, y);
  }
  function coin(x, y, r) {
    ctx.lineWidth = 2; ctx.strokeStyle = C.ink; ctx.fillStyle = "#F2C14E";
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, r * 0.62, 0, 2 * Math.PI); ctx.stroke();
  }
  function candle(x, top, w, h) {   // a red candle: the body from top down h, a wick above and below
    ctx.lineWidth = 2; ctx.strokeStyle = C.ink; ctx.fillStyle = C.red;
    ctx.beginPath(); ctx.moveTo(x + w / 2, top - h * 0.3); ctx.lineTo(x + w / 2, top + h * 1.15); ctx.stroke();
    ctx.fillRect(x, top, w, h); ctx.strokeRect(x, top, w, h);
  }
  function overlay(title, words, actions, focus) {
    $("ovTitle").textContent = title || "";
    $("ovText").textContent = words || "";
    var box = $("ovActions");
    box.innerHTML = "";
    (actions || []).forEach(function (a, i) {
      var el = document.createElement(a.href ? "a" : "button");
      el.className = "btn" + (i ? " quiet" : "");
      el.textContent = a.label;
      if (a.href) el.href = a.href; else { el.type = "button"; el.addEventListener("click", a.go); }
      box.appendChild(el);
    });
    $("overlay").hidden = !title;
    if (title && focus && box.firstChild) box.firstChild.focus({ preventScroll: true });
  }

  /* ---------- the games ---------- */
  var GAMES = {};

  GAMES.catch = {
    reset: function () { size(80); this.x = this.to = W / 2; this.dir = 0; this.items = []; this.score = 0; this.lives = 3; this.t = 0; this.next = 0.5; },
    begin: function () { say("Catch the gold. A red candle costs a heart."); },
    step: function (dt) {
      this.t += dt; this.next -= dt;
      if (this.next <= 0) {
        this.items.push({ x: rnd(20, W - 20), y: -16, v: rnd(110, 150) + Math.min(230, this.t * 4.5),
                          bad: Math.random() < Math.min(0.45, 0.22 + this.t / 250) });
        this.next = Math.max(0.3, 0.9 - this.t / 80);
      }
      if (this.dir) this.to = this.x + this.dir * 60;
      var reach = 560 * dt, top = H - 6 - FW * 1.05, chin = H - 6 - FW * 0.45;
      this.x = clamp(this.x + clamp(this.to - this.x, -reach, reach), FW / 2, W - FW / 2);
      for (var i = this.items.length - 1; i >= 0; i--) {
        var it = this.items[i];
        it.y += it.v * dt;
        if (it.y > top && it.y < chin && Math.abs(it.x - this.x) < FW * 0.42) {
          this.items.splice(i, 1);
          if (!it.bad) { this.score++; feel("happy", 450); }
          else if (--this.lives > 0) { feel("worried", 800); say(this.lives === 1 ? "One heart left." : "Ouch, a red candle."); }
          else return end(this.score, this.score + (this.score === 1 ? " coin." : " coins."));
        } else if (it.y > H + 30) this.items.splice(i, 1);
      }
      hud(this.score + " · " + "♥♥♥".slice(0, this.lives) + "♡♡♡".slice(this.lives));
    },
    draw: function () {
      this.items.forEach(function (it) { if (it.bad) candle(it.x - 8, it.y - 14, 16, 28); else coin(it.x, it.y, 11); });
      place(this.x, H - 6);
    },
    key: function (k, down) {
      var d = k === "ArrowLeft" || k === "a" ? -1 : k === "ArrowRight" || k === "d" ? 1 : 0;
      if (d && down) this.dir = d; else if (d && this.dir === d) this.dir = 0;
      return !!d;
    },
    point: function (x) { this.to = x; },
    tap: function (x) { this.to = x; }
  };

  GAMES.hop = {
    reset: function () { size(72); this.h = 0; this.v = 0; this.obs = []; this.score = 0; this.t = 0; this.next = 1; this.speed = 250; this.shift = 0; },
    begin: function () { say("Hop over the red candles. The market speeds up."); },
    step: function (dt) {
      this.t += dt;
      this.speed = Math.min(560, 250 + this.t * 7);
      this.shift = (this.shift + this.speed * dt) % 40;
      if (this.v || this.h) {
        this.v -= 1900 * dt; this.h += this.v * dt;
        if (this.h <= 0) { this.h = 0; this.v = 0; if (flip.current === "excited") flip.mood("idle"); }
      }
      this.next -= dt;
      if (this.next <= 0) {
        var h = rnd(26, 62);
        this.obs.push({ x: W + 10, w: rnd(16, 24), h: h, top: GROUND - 2 - h * 1.15, done: false });
        this.next = rnd(250, 520) / this.speed;          // never closer than a hop can clear
      }
      var fx = 110, half = FW * 0.3, feet = GROUND - this.h;
      for (var i = this.obs.length - 1; i >= 0; i--) {
        var o = this.obs[i];
        o.x -= this.speed * dt;
        if (o.x < fx + half && o.x + o.w > fx - half && feet > o.top + 3)
          return end(this.score, this.score + (this.score === 1 ? " candle" : " candles") + " cleared.");
        if (!o.done && o.x + o.w < fx - half) {
          o.done = true; this.score++;
          if (this.score % 10 === 0) { feel("happy", 600); say(this.score + " candles! Faster now."); }
        }
        if (o.x < -40) this.obs.splice(i, 1);
      }
      hud(String(this.score));
    },
    draw: function () {
      ctx.lineWidth = 2; ctx.strokeStyle = C.ink;
      ctx.beginPath(); ctx.moveTo(0, GROUND); ctx.lineTo(W, GROUND); ctx.stroke();
      ctx.strokeStyle = C.rule; ctx.beginPath();
      for (var x = -this.shift; x < W; x += 40) { ctx.moveTo(x, GROUND + 12); ctx.lineTo(x + 16, GROUND + 12); }
      ctx.stroke();
      this.obs.forEach(function (o) { candle(o.x, o.top, o.w, o.h); });
      place(110, GROUND - this.h + 2);
    },
    jump: function () { if (!this.h && !this.v) { this.v = 640; flip.mood("excited"); } },
    key: function (k, down) {
      if (k !== " " && k !== "ArrowUp" && k !== "w") return false;
      if (down) this.jump();
      return true;
    },
    tap: function () { this.jump(); }
  };

  // Edge or luck?: a record made with a real edge (small, steady, many trades) against plain luck dressed up to
  // look better: the best of 20 short coin-flip records, or a no-edge record carried by one huge trade.
  function trades(n, mu, k) { var t = []; for (var i = 0; i < n; i++) t.push(k * (mu + normal())); return t; }
  function withEdge() { var n = Math.round(rnd(220, 400)); return { t: trades(n, rnd(3.6, 4.8) / Math.sqrt(n), 0.6), tries: 1 }; }
  function bestOfTwenty(goal) {
    for (;;) {
      var n = Math.round(rnd(30, 60)), top = null;
      for (var k = 0; k < 20; k++) { var t = trades(n, 0, 1); if (!top || sum(t) > sum(top)) top = t; }
      if (sum(top) > 0) return { t: top.map(function (x) { return x * goal / sum(top); }), tries: 20 };
    }
  }
  function oneBigTrade(goal) {
    for (;;) {
      var n = Math.round(rnd(150, 260)), t = trades(n, -0.03, 0.6), big = goal - sum(t);
      if (big > goal * 0.5) { t.splice(Math.round(rnd(n * 0.3, n * 0.9)), 0, big); return { t: t, tries: 1, rest: goal - big }; }
    }
  }
  GAMES.guess = {
    reset: function () { size(56); this.round = 0; this.right = 0; this.recs = null; place(W / 2, H - 4); },
    begin: function () { this.deal(); },
    deal: function () {
      this.round++; this.shown = false;
      var edge = withEdge(), goal = sum(edge.t);
      var luck = Math.random() < 0.5 ? bestOfTwenty(goal * rnd(1.1, 1.5)) : oneBigTrade(goal * rnd(1.05, 1.35));
      this.edgeAt = Math.random() < 0.5 ? 0 : 1;
      this.recs = this.edgeAt ? [luck, edge] : [edge, luck];
      picks(true);
      feel("wait");
      say("Round " + this.round + " of 10" + (this.round > 1 ? ", " + this.right + " right so far" : "") + ". Which one has the edge?");
      draw();
    },
    pick: function (i) {
      if (this.shown) return;
      this.shown = true;
      var ok = i === this.edgeAt, luck = this.recs[1 - this.edgeAt], L = "AB"[1 - this.edgeAt], E = "AB"[this.edgeAt];
      if (ok) this.right++;
      feel(ok ? "happy" : "confused");
      say((ok ? "Right! " : "Not this time. ") + (luck.tries > 1
        ? L + " was the luckiest of " + luck.tries + " coin-flippers: " + luck.t.length + " trades and no edge at all."
        : L + " had no edge: one trade made its gain, and without it " + (luck.rest < 0 ? "it lost money." : "it made only " + Math.round(luck.rest) + "%."))
        + " " + E + " had a real one, small and steady over " + this.recs[this.edgeAt].t.length + " trades.");
      picks(false, this.round === 10 ? "See your score" : "Next round");
      draw();
    },
    next: function () {
      if (this.round < 10) this.deal();
      else end(this.right, this.right + " of 10 right." + (this.right === 10 ? " The judge would be proud." : ""));
    },
    draw: function () {
      if (!this.recs) return;
      var self = this, pw = (W - 48 - 24) / 2, top = 128, bottom = 284, lo = 0, hi = 0;
      var curves = this.recs.map(function (r) {
        var c = [0];
        r.t.forEach(function (x) { c.push(c[c.length - 1] + x); lo = Math.min(lo, c[c.length - 1]); hi = Math.max(hi, c[c.length - 1]); });
        return c;
      });
      [12, W - 12 - pw].forEach(function (x0, i) {
        var r = self.recs[i], c = curves[i], fin = c[c.length - 1], isEdge = i === self.edgeAt;
        var y = function (v) { return bottom - (v - lo) / (hi - lo || 1) * (bottom - top); };
        ctx.lineWidth = self.shown ? 3 : 1.5;
        ctx.strokeStyle = self.shown ? (isEdge ? C.green : C.amber) : C.rule;
        ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x0, 10, pw, 284, 10); else ctx.rect(x0, 10, pw, 284); ctx.stroke();
        text("AB"[i], x0 + 14, 42, 28);
        text((fin < 0 ? "" : "+") + Math.round(fin) + "%", x0 + 44, 42, 24, C["ink-2"]);
        if (self.shown) text(isEdge ? "Edge" : "Luck", x0 + pw - 14, 40, 22, isEdge ? C.green : C.amber, "right");
        text(r.t.length + " trades", x0 + 14, 66, 17, C["ink-2"]);
        text(r.tries > 1 ? "best of " + r.tries + " tries" : "1 try", x0 + 14, 86, 17, C["ink-2"]);
        text("top trade: " + Math.round(100 * Math.max.apply(null, r.t) / fin) + "% of gain", x0 + 14, 106, 17, C["ink-2"]);
        ctx.setLineDash([4, 5]); ctx.lineWidth = 1; ctx.strokeStyle = C.rule;
        ctx.beginPath(); ctx.moveTo(x0 + 10, y(0)); ctx.lineTo(x0 + pw - 10, y(0)); ctx.stroke(); ctx.setLineDash([]);
        ctx.lineWidth = 2; ctx.strokeStyle = C.ink; ctx.lineJoin = "round"; ctx.beginPath();
        c.forEach(function (v, k) { var px = x0 + 10 + k / (c.length - 1) * (pw - 20); if (k) ctx.lineTo(px, y(v)); else ctx.moveTo(px, y(v)); });
        ctx.stroke();
      });
    }
  };
  function picks(choosing, nextLabel) {
    $("picks").hidden = false;
    document.querySelectorAll("[data-pick]").forEach(function (b) { b.hidden = !choosing; });
    $("next").hidden = choosing;
    if (nextLabel) { $("next").textContent = nextLabel; $("next").focus({ preventScroll: true }); }
  }
  document.querySelectorAll("[data-pick]").forEach(function (b) { b.addEventListener("click", function () { GAMES.guess.pick(+b.getAttribute("data-pick")); }); });
  $("next").addEventListener("click", function () { GAMES.guess.next(); });

  /* ---------- playing ---------- */
  function start() {
    overlay(); colors();
    game.reset(); game.begin();
    playing = true; last = 0;
    stage.classList.toggle("live", !!game.step);
    if (game.step) requestAnimationFrame(loop);
  }
  function loop(ts) {
    if (!playing) return;
    var dt = last ? Math.min(0.05, (ts - last) / 1000) : 0;   // a hidden tab pauses, it never jumps ahead
    last = ts;
    game.step(dt);
    draw();
    if (playing) requestAnimationFrame(loop);
  }
  function end(score, line) {
    var before = best(current.id, 0), record = score > before && score > 0;
    playing = false;
    stage.classList.remove("live");
    best(current.id, score);
    feel(record ? "happy" : "meh");
    $("picks").hidden = true;
    say(record ? "A new best!" : "");
    overlay(record ? "A new best!" : "Game over", line + (record || !before ? "" : " Your best: " + before + "."),
            [{ label: "Play again", go: start }], true);
    notes();
  }
  function choose(g, first) {
    playing = false; current = g; game = GAMES[g.id];
    stage.classList.remove("live");
    document.querySelectorAll(".pick").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-id") === g.id ? "true" : "false"); });
    $("picks").hidden = true; hud(""); say(""); notes();
    if (!first && history.replaceState) history.replaceState(null, "", "#" + g.id);
    game.reset(); draw();
    var t = tier();
    if (t < g.tier) {
      var free = DATA.games.filter(function (x) { return !x.tier; })[0];
      feel("sleepy");
      overlay("A thank-you game", g.name + " opens for supporters from " + DATA.tiers[g.tier - 1] +
              (t ? " (yours here: " + DATA.tiers[t - 1] + ")" : "") + ". " + free.name + " is free for everyone.",
              [{ label: "Support edge or luck", href: "/support" }, { label: "Play " + free.name, go: function () { choose(free); } }], !first);
    } else {
      feel("idle");
      overlay(g.name, g.how, [{ label: "Play", go: start }], !first);
    }
  }
  function notes() {
    var t = tier();
    document.querySelectorAll(".pick").forEach(function (b) {
      var g = DATA.games.filter(function (x) { return x.id === b.getAttribute("data-id"); })[0], top = best(g.id, 0);
      b.querySelector("small").textContent = t < g.tier ? "🔒 From " + DATA.tiers[g.tier - 1] : top ? "Best: " + top
        : g.tier ? "Ready to play" : "Free for everyone";
    });
  }

  DATA.games.forEach(function (g) {
    var b = document.createElement("button"), name = document.createElement("b");
    b.type = "button"; b.className = "pick"; b.setAttribute("data-id", g.id);
    name.textContent = g.name;
    b.appendChild(name); b.appendChild(document.createElement("small"));
    b.addEventListener("click", function () { choose(g); });
    $("picker").appendChild(b);
  });
  notes();

  function onKey(e, down) {
    if (!playing || !game.key || e.ctrlKey || e.metaKey || e.altKey) return;
    if (game.key(e.key, down)) e.preventDefault();
  }
  function worldX(e) { var r = stage.getBoundingClientRect(); return (e.clientX - r.left) / r.width * W; }
  document.addEventListener("keydown", function (e) { onKey(e, true); });
  document.addEventListener("keyup", function (e) { onKey(e, false); });
  stage.addEventListener("pointermove", function (e) { if (playing && game.point) game.point(worldX(e)); });
  stage.addEventListener("pointerdown", function (e) { if (playing && game.tap && !e.target.closest(".btn")) game.tap(worldX(e)); });
  document.addEventListener("eol:theme", function () { colors(); draw(); });
  window.addEventListener("resize", fit);

  colors();
  var wanted = DATA.games.filter(function (g) { return "#" + g.id === location.hash; })[0];
  var open = DATA.games.filter(function (g) { return g.tier <= tier(); });
  fit();
  choose(wanted || open[open.length - 1] || DATA.games[0], true);
  window.EOLGames = { choose: function (id) { choose(DATA.games.filter(function (g) { return g.id === id; })[0]); }, start: start,
                      state: function () { return game; } };   // for screenshots and tests
})();
