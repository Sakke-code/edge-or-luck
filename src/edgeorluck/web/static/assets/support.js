/* edge or luck: the support page. Flip's wardrobe: pick what he wears from what this browser has unlocked (a
   supporter's thank-you link sets it, see support.py). Saved in this browser only; every page's Flip wears it. */
(function () {
  "use strict";
  var W = window.EOL_WARDROBE, $ = function (id) { return document.getElementById(id); };
  function tier() { try { return +localStorage.getItem("eol-tier") || 0; } catch (e) { return 0; } }
  var outfit = EOL.storedOutfit();
  var big = EOL.makeFlip($("wardrobePet"), null, outfit);
  big.mood("happy");
  document.querySelectorAll(".tier-pet").forEach(function (host) {
    EOL.makeFlip(host, null, JSON.parse(host.getAttribute("data-outfit"))).mood("idle");
  });

  function save() {
    try { localStorage.setItem("eol-flip", JSON.stringify(outfit)); } catch (e) {}
    big.wear(outfit);
    document.dispatchEvent(new CustomEvent("eol:outfit"));
  }

  function render() {
    var t = tier(), box = $("slots");
    box.innerHTML = "";
    $("tierNow").textContent = t
      ? "Unlocked here: " + W.tiers[t - 1] + ". Tap something to put it on, and again to take it off."
      : "Unlocked here: nothing yet. Support, then open the thank-you link you get.";
    W.slots.forEach(function (s) {
      var row = document.createElement("div"), h = document.createElement("h3");
      row.className = "slot";
      h.textContent = s[1];
      row.appendChild(h);
      W.items.filter(function (i) { return i.slot === s[0]; }).forEach(function (i) {
        var b = document.createElement("button"), locked = i.tier > t;
        b.type = "button";
        b.className = "item";
        b.disabled = locked;
        b.setAttribute("aria-pressed", outfit[i.slot] === i.id ? "true" : "false");
        b.textContent = i.name + (locked ? " · " + W.tiers[i.tier - 1] : "");
        b.addEventListener("click", function () {
          outfit[i.slot] = outfit[i.slot] === i.id ? null : i.id;
          save();
          render();
          if (outfit[i.slot] && (i.slot === "emote" || i.slot === "move")) big.cheer();
        });
        row.appendChild(b);
      });
      box.appendChild(row);
    });
  }

  render();
})();
