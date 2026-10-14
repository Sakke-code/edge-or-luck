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
    '</g></svg>';
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
    confused: ["eyes", "b-conf", "m-flat", "a-down", "x-q"]
  };
  function makeFlip(host, bubble) {
    host.innerHTML = FLIP_SVG;
    var svg = host.querySelector("svg"), eyes = svg.querySelector(".eyes"), words = bubble && bubble.querySelector(".say");
    var api = {
      svg: svg, current: null,
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

  window.EOL = { makeFlip: makeFlip, cssVar: cssVar, isDark: isDark, reduce: reduce };
})();
