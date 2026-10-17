(async () => {
  const m = await (await fetch("manifest.json")).json();
  const $ = (id) => document.getElementById(id);
  const votes = JSON.parse(localStorage.getItem("bntts_votes") || "[]");
  let cur = null;
  const pick = (xs) => xs[Math.floor(Math.random() * xs.length)];
  function round() {
    const s = pick(m.sentences.filter((x) => Object.keys(x.clips).length >= 2));
    const sys = Object.keys(s.clips).sort(() => Math.random() - 0.5).slice(0, 2);
    cur = { id: s.id, A: sys[0], B: sys[1] };
    $("meta").textContent = `${s.category} · ${s.id}`;
    $("sent").textContent = s.text;
    $("a").src = s.clips[sys[0]]; $("b").src = s.clips[sys[1]];
    $("reveal").textContent = "";
  }
  function tally() {
    const w = {};
    for (const v of votes) {
      for (const k of [v.A, v.B]) w[k] = w[k] || { win: 0, n: 0 };
      w[v.A].n++; w[v.B].n++;
      if (v.vote === "A") w[v.A].win++; else if (v.vote === "B") w[v.B].win++;
      else { w[v.A].win += 0.5; w[v.B].win += 0.5; }
    }
    const rows = Object.entries(w).sort((a, b) => b[1].win / b[1].n - a[1].win / a[1].n);
    $("tally").innerHTML = "<tr><th>system</th><th>win rate</th><th>rounds</th></tr>" +
      rows.map(([k, v]) => `<tr><td>${m.labels[k] || k}</td><td>${(100 * v.win / v.n).toFixed(0)}%</td><td>${v.n}</td></tr>`).join("");
  }
  document.querySelectorAll("[data-v]").forEach((b) => b.onclick = () => {
    if (!cur) return;
    votes.push({ ...cur, vote: b.dataset.v, t: Date.now() });
    localStorage.setItem("bntts_votes", JSON.stringify(votes));
    $("reveal").textContent = `A = ${m.labels[cur.A] || cur.A}, B = ${m.labels[cur.B] || cur.B}`;
    cur = null; tally(); setTimeout(round, 1800);
  });
  $("next").onclick = round;
  $("export").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(votes, null, 1)], { type: "application/json" }));
    a.download = "bangla-tts-votes.json"; a.click();
  };
  $("reset").onclick = () => { votes.length = 0; localStorage.removeItem("bntts_votes"); tally(); };
  $("grid").innerHTML = m.sentences.map((s) => `<details><summary lang="bn">${s.id} · ${s.text}</summary>` +
    Object.entries(s.clips).map(([k, u]) => `<div>${m.labels[k] || k}<audio controls preload="none" src="${u}"></audio></div>`).join("") +
    "</details>").join("");
  round(); tally();
})();
