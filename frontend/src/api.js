const API = "http://localhost:8000";

export async function getBuys() {
  const res = await fetch(`${API}/flips/buy-recommendations`);
  return res.json();
}

export async function getActive() {
  const res = await fetch(`${API}/flips/active`);
  return res.json();
}

export async function addActive(item_id) {
  await fetch(`${API}/flips/add/${item_id}`, { method: "POST" });
}

export async function removeActive(item_id) {
  await fetch(`${API}/flips/remove/${item_id}`, { method: "DELETE" });
}

export async function getSellSignals() {
  const res = await fetch(`${API}/flips/sell-signals`);
  return res.json();
}
