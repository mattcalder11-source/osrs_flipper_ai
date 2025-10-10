const API = "http://209.38.91.166:8000";

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

export async function closeFlip(item_id) {
  try {
    const res = await fetch(`${API_BASE}/flips/close/${item_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) throw new Error(`Failed to close flip: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error("❌ closeFlip error:", err);
    return null;
  }
}