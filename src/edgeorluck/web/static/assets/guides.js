/* edge or luck: the guides. Wakes up every Flip on the page, makes them look at the
   pointer and hop when they scroll into view, and gives code blocks a copy button.
   The text is all in the HTML: without this script the guides read the same. */
(function () {
  "use strict";
  var pets = [];
  document.querySelectorAll("[data-mood]").forEach(function (host) {
    var pet = EOL.makeFlip(host, null);
    pet.mood(host.getAttribute("data-mood") || "idle");
    pets.push({ host: host, pet: pet });
    host.addEventListener("click", function () { pet.hop(); });
  });

  if (!EOL.reduce) {
    var px = 0, py = 0, queued = false;
    window.addEventListener("pointermove", function (e) {
      px = e.clientX; py = e.clientY;
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { queued = false; pets.forEach(function (p) { p.pet.look(px, py); }); });
    }, { passive: true });
    if ("IntersectionObserver" in window) {
      var seen = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          var p = pets.filter(function (q) { return q.host === en.target; })[0];
          if (p) p.pet.hop();
          seen.unobserve(en.target);
        });
      }, { threshold: 0.8 });
      document.querySelectorAll(".flip-says [data-mood]").forEach(function (h) { seen.observe(h); });
    }
  }

  document.querySelectorAll(".code .copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.parentNode.querySelector("pre").textContent;
      function done(ok) {
        btn.textContent = ok ? "Copied" : "Select and copy";
        setTimeout(function () { btn.textContent = "Copy"; }, 1600);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
      } else {
        done(false);
      }
    });
  });
})();
