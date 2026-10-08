// Playground: edit a state and typed questions, send them to the s1 Space through @gradio/client, render answers.
const cfg = window.S1_PLAYGROUND;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let examples = [];
let questions = {};          // name -> {type, instructions, criteria}
let mode = "form";
let client = null;

// ---- question builder ----------------------------------------------------------------------------------------

function blankQuestion(type) {
  if (type === "choice") return { type, instructions: "", criteria: { option_a: "", option_b: "" } };
  if (type === "score") return { type, instructions: "", criteria: ["low", "medium", "high"] };
  return { type: "noul", instructions: "" };
}

function renderBuilder() {
  const root = $("pg-builder");
  root.innerHTML = "";
  Object.entries(questions).forEach(([name, q], idx) => {
    const card = document.createElement("div");
    card.className = "pg-qcard";
    let opts = "";
    if (q.type === "choice") {
      opts = Object.entries(q.criteria || {}).map(([k, d], i) => `
        <div class="pg-opt" data-i="${i}">
          <input class="pg-in pg-key" value="${esc(k)}" aria-label="Option key" placeholder="key">
          <input class="pg-in pg-desc" value="${esc(d ?? "")}" aria-label="Option description" placeholder="description">
          <button type="button" class="pg-x" aria-label="Remove option">&times;</button>
        </div>`).join("") + `<button type="button" class="pg-link pg-addopt">+ option</button>`;
    } else if (q.type === "score") {
      opts = (q.criteria || []).map((lvl, i) => `
        <div class="pg-opt" data-i="${i}"><span class="pg-lvl">${i}</span>
          <input class="pg-in pg-desc" value="${esc(lvl)}" aria-label="Level ${i}">
          <button type="button" class="pg-x" aria-label="Remove level">&times;</button>
        </div>`).join("") + `<button type="button" class="pg-link pg-addopt">+ level (lowest first)</button>`;
    } else {
      opts = `<p class="pg-hint">Answers yes or no, with the probability of yes.</p>`;
    }
    card.innerHTML = `
      <div class="pg-qrow">
        <input class="pg-in pg-name" value="${esc(name)}" aria-label="Question name">
        <select class="pg-in pg-type" aria-label="Question type">
          <option value="choice"${q.type === "choice" ? " selected" : ""}>choice</option>
          <option value="noul"${q.type === "noul" ? " selected" : ""}>yes / no</option>
          <option value="score"${q.type === "score" ? " selected" : ""}>score</option>
        </select>
        <button type="button" class="pg-x pg-del" aria-label="Remove question">&times;</button>
      </div>
      <textarea class="pg-in pg-instr" rows="2" aria-label="Instructions" placeholder="What should s1 decide?">${esc(q.instructions)}</textarea>
      <div class="pg-opts">${opts}</div>`;
    card.dataset.name = name;
    card.dataset.idx = idx;
    root.appendChild(card);
  });
}

function readBuilder() {
  const out = {};
  for (const card of $("pg-builder").querySelectorAll(".pg-qcard")) {
    const name = card.querySelector(".pg-name").value.trim() || `q${Object.keys(out).length + 1}`;
    const type = card.querySelector(".pg-type").value;
    const q = { type, instructions: card.querySelector(".pg-instr").value.trim() };
    if (type === "choice") {
      q.criteria = {};
      card.querySelectorAll(".pg-opt").forEach((o) => {
        const k = o.querySelector(".pg-key").value.trim();
        if (k) q.criteria[k] = o.querySelector(".pg-desc").value.trim();
      });
    } else if (type === "score") {
      q.criteria = [...card.querySelectorAll(".pg-opt .pg-desc")].map((i) => i.value.trim()).filter(Boolean);
    }
    out[name] = q;
  }
  return out;
}

$("pg-builder").addEventListener("change", (e) => {
  if (e.target.classList.contains("pg-type")) {
    questions = readBuilder();
    const card = e.target.closest(".pg-qcard");
    const name = card.querySelector(".pg-name").value.trim();
    questions[name] = { ...blankQuestion(e.target.value), instructions: questions[name].instructions };
    renderBuilder();
  }
});
$("pg-builder").addEventListener("click", (e) => {
  const card = e.target.closest(".pg-qcard");
  if (!card) return;
  if (e.target.classList.contains("pg-del")) { card.remove(); questions = readBuilder(); renderBuilder(); }
  else if (e.target.classList.contains("pg-x")) { e.target.closest(".pg-opt").remove(); questions = readBuilder(); renderBuilder(); }
  else if (e.target.classList.contains("pg-addopt")) {
    questions = readBuilder();
    const q = questions[card.querySelector(".pg-name").value.trim()];
    if (q.type === "choice") q.criteria[`option_${Object.keys(q.criteria).length + 1}`] = "";
    else q.criteria.push(`level ${q.criteria.length}`);
    renderBuilder();
  }
});
$("pg-add").addEventListener("click", () => {
  questions = readBuilder();
  let n = Object.keys(questions).length + 1;
  while (questions[`q${n}`]) n++;
  questions[`q${n}`] = blankQuestion("choice");
  renderBuilder();
});

function setMode(m) {
  if (m === mode) return;
  if (m === "json") { questions = readBuilder(); $("pg-questions").value = JSON.stringify(questions, null, 2); }
  else {
    try { questions = JSON.parse($("pg-questions").value); $("pg-q-err").textContent = ""; }
    catch (err) { $("pg-q-err").textContent = `Questions are not valid JSON: ${err.message}`; return; }
    renderBuilder();
  }
  mode = m;
  $("pg-builder").hidden = $("pg-add").hidden = m === "json";
  $("pg-questions").hidden = m !== "json";
  $("pg-mode-form").setAttribute("aria-selected", String(m === "form"));
  $("pg-mode-json").setAttribute("aria-selected", String(m === "json"));
}
$("pg-mode-form").addEventListener("click", () => setMode("form"));
$("pg-mode-json").addEventListener("click", () => setMode("json"));

// ---- examples ------------------------------------------------------------------------------------------------

function loadExample(i) {
  const ex = examples[i];
  $("pg-state").value = ex.state;
  questions = JSON.parse(ex.questions);
  if (mode === "json") $("pg-questions").value = JSON.stringify(questions, null, 2); else renderBuilder();
  document.querySelectorAll(".pg-chip").forEach((c, j) => c.setAttribute("aria-pressed", String(i === j)));
  $("pg-answers").innerHTML = '<p class="pg-empty">Run a request to see typed answers here.</p>';
  $("pg-latency").textContent = "";
  updateCode();
}

// ---- request -------------------------------------------------------------------------------------------------

function currentQuestions() {
  if (mode === "json") return JSON.parse($("pg-questions").value);
  return readBuilder();
}

function updateCode() {
  let q;
  try { q = currentQuestions(); } catch { return; }
  $("pg-code").textContent =
`# pip install gradio_client
from gradio_client import Client

client = Client("${cfg.space}")
result = client.predict(
    state_text=${JSON.stringify($("pg-state").value)},
    questions_text=${JSON.stringify(JSON.stringify(q))},
    api_name="/decide",
)
print(result["answers"])`;
}

function bar(label, p, top) {
  return `<div class="pg-row${top ? " pg-top" : ""}"><span class="pg-lab" title="${esc(label)}">${esc(label)}</span>`
    + `<span class="pg-track"><span class="pg-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>`
    + `<span class="pg-p">${p.toFixed(2)}</span></div>`;
}

function renderAnswers(res, qs) {
  if (res.error) { $("pg-answers").innerHTML = `<div class="pg-errbox">${esc(res.error)}</div>`; return; }
  const cards = Object.entries(res.answers).map(([name, a]) => {
    const q = qs[name] || {};
    let head, rows;
    if (a.type === "noul") {
      const yes = a.noul >= 0.5;
      head = `${yes ? "Yes" : "No"} <small>P(yes) ${a.noul.toFixed(2)}</small>`;
      rows = bar("yes", a.noul, yes) + bar("no", 1 - a.noul, !yes);
    } else if (a.type === "choice") {
      const desc = (q.criteria || {})[a.choice];
      head = `${esc(a.choice)}${desc ? ` <span class="pg-desc-inline">${esc(desc)}</span>` : ""} <small>${a.confidence.toFixed(2)}</small>`;
      rows = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).map(([k, p]) => bar(k, p, k === a.choice)).join("");
    } else {
      const best = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])[0][0];
      head = `${esc(a.legend[best])} <small>expected level ${a.score.toFixed(2)}</small>`;
      rows = Object.entries(a.probabilities).map(([k, p]) => bar(`${k} · ${a.legend[k]}`, p, k === best)).join("");
    }
    return `<article class="pg-acard"><div class="pg-aq"><code>${esc(name)}</code> <span class="pg-tag">${esc(a.type === "noul" ? "yes/no" : a.type)}</span> ${esc(q.instructions || "")}</div>`
      + `<div class="pg-ahead">${head}</div>${rows}</article>`;
  });
  $("pg-answers").innerHTML = cards.join("");
  $("pg-latency").textContent = `${res.usage.questions} question${res.usage.questions === 1 ? "" : "s"} · ${Math.round(res.usage.latency_ms)} ms server time`;
}

async function run() {
  $("pg-state-err").textContent = $("pg-q-err").textContent = "";
  let qs;
  try { qs = currentQuestions(); } catch (err) { $("pg-q-err").textContent = `Questions are not valid JSON: ${err.message}`; return; }
  const state = $("pg-state").value;
  if (/^\s*[{[]/.test(state)) { try { JSON.parse(state); } catch (err) { $("pg-state-err").textContent = `State looks like JSON but does not parse (${err.message}); it will be sent as text.`; } }
  const btn = $("pg-run");
  btn.disabled = true;
  const t0 = performance.now();
  try {
    $("pg-status").textContent = "Connecting to the Space…";
    if (!client) {
      const { Client } = await import(cfg.client);
      client = await Client.connect(cfg.space);
    }
    const job = client.submit("/decide", { state_text: state, questions_text: JSON.stringify(qs) });
    let result = null;
    for await (const msg of job) {
      if (msg.type === "status") {
        if (msg.stage === "pending") $("pg-status").textContent = msg.queue && msg.position != null ? `In queue, position ${msg.position + 1}…` : "Waiting for a GPU…";
        else if (msg.stage === "error") throw new Error(msg.message || "the Space returned an error");
      } else if (msg.type === "data") { result = msg.data[0]; break; }
    }
    renderAnswers(result, qs);
    $("pg-raw").textContent = JSON.stringify(result, null, 2);
    $("pg-status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (err) {
    client = null;
    const msg = String(err && err.message || err);
    $("pg-status").textContent = "";
    $("pg-answers").innerHTML = `<div class="pg-errbox">${/quota/i.test(msg)
      ? "The free GPU quota for this browser is used up for today. Sign in to Hugging Face, or try again later."
      : `The Space did not answer: ${esc(msg)}. It may be starting up; try again in a minute.`}</div>`;
  } finally {
    btn.disabled = false;
  }
}
$("pg-run").addEventListener("click", run);
$("pg-state").addEventListener("input", updateCode);
$("pg-builder").addEventListener("input", updateCode);
$("pg-questions").addEventListener("input", updateCode);

// ---- start ---------------------------------------------------------------------------------------------------

fetch("assets/examples.json").then((r) => r.json()).then((list) => {
  examples = list;
  $("pg-examples").innerHTML = list.map((e, i) => `<button type="button" class="pg-chip" aria-pressed="false" data-i="${i}">${esc(e.name)}</button>`).join("");
  $("pg-examples").addEventListener("click", (e) => { const b = e.target.closest(".pg-chip"); if (b) loadExample(Number(b.dataset.i)); });
  loadExample(0);
});
