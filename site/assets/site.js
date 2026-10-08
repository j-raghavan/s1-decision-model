// Table sorting and quick-start tabs. No dependencies; the pages work without it.
(function () {
  "use strict";

  function sortTable(table, th) {
    var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
    var numeric = th.getAttribute("data-sort") === "num";
    var current = th.getAttribute("aria-sort");
    // numbers default to descending except columns where lower is better (ECE, MAE): ascending first
    var lowerBetter = /lower is better/i.test(th.textContent);
    var dir = current ? (current === "descending" ? "ascending" : "descending")
                      : (numeric && !lowerBetter ? "descending" : "ascending");
    table.querySelectorAll("th[data-sort]").forEach(function (h) { h.removeAttribute("aria-sort"); });
    th.setAttribute("aria-sort", dir);
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var x = a.children[idx].getAttribute("data-v") || a.children[idx].textContent;
      var y = b.children[idx].getAttribute("data-v") || b.children[idx].textContent;
      var c = numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y);
      return dir === "ascending" ? c : -c;
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }

  document.querySelectorAll("table.sortable").forEach(function (table) {
    table.querySelectorAll("th[data-sort]").forEach(function (th) {
      th.tabIndex = 0;
      th.addEventListener("click", function () { sortTable(table, th); });
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortTable(table, th); }
      });
    });
  });

  document.querySelectorAll("[data-tabs]").forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
      tab.focus();
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight") select(tabs[(i + 1) % tabs.length]);
        if (e.key === "ArrowLeft") select(tabs[(i - 1 + tabs.length) % tabs.length]);
      });
    });
  });
})();
