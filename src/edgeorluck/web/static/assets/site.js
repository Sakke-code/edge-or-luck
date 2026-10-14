/* edge or luck: what every page shares: the paper/blueprint switch and Flip, the referee coin.
   Load it after the header (it wires the switch) and before the page's own script. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- theme ---------- */
  var root = document.documentElement;
  function storedTheme() { try { return localStorage.getItem("eol-theme"); } catch (e) { return null; } }
  function isDark() {
    var t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncThemeButton() {
    var dark = isDark();
    $("themeLabel").textContent = dark ? "Paper" : "Blueprint";
    $("theme").setAttribute("aria-label", dark ? "Switch to paper (light) look" : "Switch to blueprint (dark) look");
  }
  var t0 = storedTheme();
  if (t0 === "dark" || t0 === "light") root.setAttribute("data-theme", t0);
  syncThemeButton();
  $("theme").addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("eol-theme", next); } catch (e) {}
    syncThemeButton();
    document.dispatchEvent(new CustomEvent("eol:theme"));
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { syncThemeButton(); document.dispatchEvent(new CustomEvent("eol:theme")); });

  function cssVar(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }

  var FLIP_SVG = '<svg class="flip" viewBox="-42 -50 84 90" aria-hidden="true"><g class="all">' +
    '<path class="ln leg-l" d="M-8 20.5-10 35.5h-5.5"/><path class="ln leg-r" d="M8 20.5l2 15h5.5"/>' +
    '<circle class="coin" cx="0" cy="-4" r="25"/><circle class="rim" cx="0" cy="-4" r="19.5"/>' +
    '<g class="eyes" data-p="eyes"><g class="pupils"><ellipse class="ink" cx="-8" cy="-9" rx="2.7" ry="3.3"/><ellipse class="ink" cx="8" cy="-9" rx="2.7" ry="3.3"/></g></g>' +
    '<path data-p="e-happy" class="ln" d="M-11.5-8.5q3.5-5 7 0M4.5-8.5q3.5-5 7 0"/>' +
    '<path data-p="e-sleep" class="ln" d="M-11.5-8h7M4.5-8h7"/>' +
    '<path data-p="b-up" class="ln" d="M-12-16.5q3.5-3 7 0M5-16.5q3.5-3 7 0"/>' +
    '<path data-p="b-worry" class="ln" d="M-12-14.5-5-17.5M12-14.5 5-17.5"/>' +
    '<path data-p="b-conf" class="ln" d="M-12-16q3.5-3.5 7 0M5-15h7"/>' +
    '<path data-p="m-smile" class="ln" d="M-5.5 2.5q5.5 4.5 11 0"/>' +
    '<path data-p="m-grin" class="ink" style="stroke:currentColor;stroke-width:1.4;stroke-linejoin:round" d="M-8 1.5q8 9.5 16 0-8 2.6-16 0z"/>' +
    '<path data-p="m-flat" class="ln" d="M-5.5 4h11"/>' +
    '<path data-p="m-wavy" class="ln" d="M-7 5q1.75-2.6 3.5 0t3.5 0 3.5 0 3.5 0"/>' +
    '<ellipse data-p="m-o" class="ln" cx="0" cy="4" rx="2.8" ry="3.4"/>' +
    '<path data-p="m-sleep" class="ln" d="M-4 4.5q4 2 8 0"/>' +
    '<path data-p="a-down" class="ln" d="M-24.5 1q-7 5-8.5 13M24.5 1q7 5 8.5 13"/>' +
    '<path data-p="a-up" class="ln" d="M-24-3q-8-1-11.5-11M24-3q8-1 11.5-11"/>' +
    '<path data-p="a-toss" class="ln" d="M-24.5 1q-7 5-8.5 13M24-3q7-3 9-14"/>' +
    '<ellipse data-p="x-coin" class="tiny tossme" cx="34" cy="-21" rx="4.6" ry="4.6"/>' +
    '<path data-p="x-spark" class="spark" d="M-33-33v6M-36-30h6M34-38v5M31.5-35.5h5M30-15v4M28-13h4"/>' +
    '<path data-p="x-sweat" class="drop-sweat" d="M20-30q3.6 4.6 0 7.2-3.6-2.6 0-7.2z"/>' +
    '<g data-p="x-zz"><text class="hand-t" x="20" y="-30" font-size="10">z</text><text class="hand-t" x="27" y="-39" font-size="13">z</text></g>' +
    '<text data-p="x-q" class="hand-t" x="22" y="-31" font-size="17">?</text>' +
    '<g data-p="x-dots"><circle class="ink" cx="24" cy="-33" r="1.6"/><circle class="ink" cx="30" cy="-37" r="1.9"/><circle class="ink" cx="37" cy="-41.5" r="2.2"/></g>' +
    // supporters' faces and moves
    '<g data-p="e-wink"><path class="ln" d="M-11.5-9q3.5 2.5 7 0"/><ellipse class="ink" cx="8" cy="-9" rx="2.7" ry="3.3"/></g>' +
    '<g data-p="e-stars"><path class="acc gold" d="M-8-14.5l1.6 3.4 3.7.4-2.8 2.5.8 3.7-3.3-1.9-3.3 1.9.8-3.7-2.8-2.5 3.7-.4zM8-14.5l1.6 3.4 3.7.4-2.8 2.5.8 3.7-3.3-1.9-3.3 1.9.8-3.7-2.8-2.5 3.7-.4z"/></g>' +
    '<g data-p="e-hearts"><path class="acc red" d="M-8-5.5q-6-3.8-6-7.2 0-2.8 3-2.8 2 0 3 2 1-2 3-2 3 0 3 2.8 0 3.4-6 7.2zM8-5.5q-6-3.8-6-7.2 0-2.8 3-2.8 2 0 3 2 1-2 3-2 3 0 3 2.8 0 3.4-6 7.2z"/></g>' +
    '<g data-p="a-wave"><path class="ln" d="M-24.5 1q-7 5-8.5 13"/><path class="ln waver" d="M24-3q8-1 11.5-11"/></g>' +
    '<g data-p="x-juggle"><circle class="tiny j1" cx="0" cy="-36" r="4"/><circle class="tiny j2" cx="0" cy="-36" r="4"/><circle class="tiny j3" cx="0" cy="-36" r="4"/></g>' +
    '</g></svg>';

  /* ---------- Flip's wardrobe: thank-you outfits for supporters (cosmetic only, the judge is the same for everyone) ---------- */
  var WEAR = {
    party: '<g transform="rotate(14 0 -27)"><path class="acc amber" d="M-9-26.5 0-48l9 21.5q-9 3-18 0z"/><path class="acc-ln" d="M-5.6-35.2l9.4 2.4M-3.2-41.3l5.3 1.4"/><circle class="acc red" cx="0" cy="-49" r="3"/></g>',
    cap: '<path class="acc violet" d="M-18-23q0-13 18-13t18 13q-18 3-36 0z"/><path class="acc violet" d="M9-23q14-3 21 2-10 3-21 .5z"/><circle class="ink" cx="0" cy="-36" r="1.7"/>',
    tophat: '<path class="acc dark" d="M-11-28v-21h22v21"/><path class="acc red" d="M-11-34h22v4.5h-22z"/><ellipse class="acc dark" cx="0" cy="-27.5" rx="18" ry="3.3"/>',
    crown: '<path class="acc gold" d="M-14-26v-14l7 7 7-12 7 12 7-7v14q-14 3-28 0z"/><circle class="acc red" cx="0" cy="-31" r="2"/><circle class="acc violet" cx="-8.5" cy="-29.5" r="1.4"/><circle class="acc violet" cx="8.5" cy="-29.5" r="1.4"/>',
    wizard: '<path class="acc violet" d="M-16-26q6-5 10-13 7-12 20-14-10 7-8 15 2 7 10 12-16 4-32 0z"/><path class="acc gold" d="M1-38l1.2 2.6 2.8.3-2.1 1.9.6 2.8-2.5-1.4-2.5 1.4.6-2.8-2.1-1.9 2.8-.3z"/>',
    shades: '<path class="ink" d="M-14.5-12.5h11v5q0 4.5-5.5 4.5t-5.5-4.5zM3.5-12.5h11v5q0 4.5-5.5 4.5t-5.5-4.5z"/><path class="acc-ln" d="M-3.5-11h7M-14.5-11.5l-8.5-2.5M14.5-11.5l8.5-2.5"/>',
    bowtie: '<path class="acc red" d="M0 14.5-7.5 10v9zM0 14.5l7.5-4.5v9z"/><circle class="acc red" cx="0" cy="14.5" r="2.1"/>',
    scarf: '<path class="acc red" d="M-17 12q17 8 34 0l.8 5.5q-17.8 8.5-35.6 0z"/><path class="acc red" d="M9 17.5l2.5 13 5.5-1.5-2.8-12.5z"/>',
    cape: '<path class="acc red" d="M-19-14q-13 28-11 50l15-4 15 5 15-5 15 4q2-22-11-50z"/>'
  };
  var SLOTS = ["back", "hat", "face", "neck"];
  var SLOT_OF = { party: "hat", cap: "hat", tophat: "hat", crown: "hat", wizard: "hat", shades: "face", bowtie: "neck", scarf: "neck", cape: "back" };
  function storedOutfit() { try { return JSON.parse(localStorage.getItem("eol-flip") || "{}") || {}; } catch (e) { return {}; } }
  function wear(svg, outfit) {
    var all = svg.querySelector(".all");
    svg.querySelectorAll("[data-wear]").forEach(function (n) { n.remove(); });
    SLOTS.forEach(function (slot) {
      var id = outfit[slot];
      if (!id || !WEAR[id] || SLOT_OF[id] !== slot) return;
      var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("data-wear", slot);
      g.innerHTML = WEAR[id];
      if (slot === "back") all.insertBefore(g, all.firstChild); else all.appendChild(g);
    });
    svg.classList.toggle("skin-gold", outfit.skin === "gold");
    svg.classList.toggle("skin-rainbow", outfit.skin === "rainbow");
    if (outfit.skin === "rainbow" && !document.getElementById("eol-rainbow")) {
      var defs = document.createElement("div");
      defs.innerHTML = '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><linearGradient id="eol-rainbow" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#E5484D"/><stop offset=".25" stop-color="#F5A524"/><stop offset=".5" stop-color="#30A46C"/><stop offset=".75" stop-color="#3E63DD"/><stop offset="1" stop-color="#8E4EC6"/></linearGradient></defs></svg>';
      document.body.appendChild(defs.firstChild);
    }
  }
  var MOVES = { wave: 1800, dance: 2000, spin: 1600, juggle: 2600 };
  var MOODS = {
    idle: ["eyes", "m-smile", "a-down"],
    excited: ["eyes", "b-up", "m-o", "a-up"],
    flipping: ["eyes", "m-o", "a-toss", "x-coin"],
    happy: ["e-happy", "m-grin", "a-up", "x-spark"],
    shrug: ["eyes", "b-up", "m-flat", "a-up"],
    meh: ["eyes", "m-flat", "a-down"],
    worried: ["eyes", "b-worry", "m-wavy", "a-down", "x-sweat"],
    sleepy: ["e-sleep", "m-sleep", "a-down", "x-zz"],
    wait: ["eyes", "m-smile", "a-down", "x-dots"],
    confused: ["eyes", "b-conf", "m-flat", "a-down", "x-q"],
    wink: ["e-wink", "m-grin", "a-up"],
    stars: ["e-stars", "m-o", "a-up", "x-spark"],
    hearts: ["e-hearts", "m-smile", "a-down"],
    wave: ["eyes", "m-smile", "a-wave"],
    juggle: ["eyes", "m-o", "a-up", "x-juggle"]
  };
  // outfit: a given outfit (the wardrobe's preview), else the one this browser saved
  function makeFlip(host, bubble, outfit) {
    host.innerHTML = FLIP_SVG;
    var svg = host.querySelector("svg"), eyes = svg.querySelector(".eyes"), words = bubble && bubble.querySelector(".say");
    var own = !outfit, dress = outfit || storedOutfit(), busy = false;
    var api = {
      svg: svg, current: null,
      wear: function (o) { dress = o || {}; wear(svg, dress); },
      // A supporter's chosen face and move, played when Flip is tapped
      cheer: function () {
        if (busy || (!dress.emote && !dress.move)) return;
        busy = true;
        var before = api.current, move = dress.move && !reduce ? dress.move : null;
        if (dress.emote) api.mood(dress.emote);
        if (move === "wave" || move === "juggle") api.mood(move);
        if (move) svg.classList.add("move-" + move);
        setTimeout(function () {
          if (move) svg.classList.remove("move-" + move);
          api.mood(before || "idle");
          busy = false;
        }, move ? MOVES[move] : 1600);
      },
      mood: function (m) {
        var parts = MOODS[m] || MOODS.idle;
        svg.querySelectorAll("[data-p]").forEach(function (p) { p.style.display = parts.indexOf(p.getAttribute("data-p")) >= 0 ? "" : "none"; });
        svg.classList.toggle("bounce", m === "excited");
        svg.classList.toggle("tossing", m === "flipping");
        if (m === "happy" && api.current !== "happy") api.hop();
        api.current = m;
      },
      say: function (t) {
        if (!words || words.textContent === t) return;
        words.textContent = t;
        bubble.classList.remove("pop"); void bubble.offsetWidth; bubble.classList.add("pop");
      },
      hop: function () {
        if (reduce) return;
        svg.classList.remove("hop"); svg.getBoundingClientRect(); svg.classList.add("hop");
      },
      look: function (x, y) {
        var r = svg.getBoundingClientRect();
        if (!r.width || r.bottom < 0 || r.top > window.innerHeight) return;
        var dx = x - (r.left + r.width / 2), dy = y - (r.top + r.height * 0.38), d = Math.sqrt(dx * dx + dy * dy) || 1;
        var k = Math.min(1, d / 200) * 2.3;
        eyes.setAttribute("transform", "translate(" + (dx / d * k).toFixed(2) + " " + (dy / d * k).toFixed(2) + ")");
      }
    };
    svg.addEventListener("animationend", function (e) { if (e.animationName === "hop") svg.classList.remove("hop"); });
    svg.addEventListener("click", function () { api.cheer(); });
    wear(svg, dress);
    if (own) document.addEventListener("eol:outfit", function () { api.wear(storedOutfit()); });
    api.mood("idle");
    if (!reduce) (function blink() {
      setTimeout(function () {
        if (api.current !== "flipping" && api.current !== "happy" && api.current !== "sleepy") {
          svg.classList.add("blink");
          setTimeout(function () { svg.classList.remove("blink"); }, 140);
        }
        blink();
      }, 2600 + Math.random() * 3800);
    })();
    return api;
  }

  window.EOL = { makeFlip: makeFlip, cssVar: cssVar, isDark: isDark, reduce: reduce, storedOutfit: storedOutfit };
})();
