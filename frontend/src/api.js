// src/api.js
const API_BASE = "http://209.38.91.166:8000";

/**
 * Fetch BUY recommendations
 */
export async function getBuys() {
  try {
    const res = await fetch(`${API_BASE}/flips/buy-recommendations`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    // ✅ Return only the array of flips
    return json.data || [];
  } catch (err) {
    console.error("❌ getBuys failed:", err);
    return [];
  }
}

/**
 * Fetch active flips
 */
export async function getActive() {
  try {
    const res = await fetch(`${API_BASE}/flips/active`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return json.data || [];
  } catch (err) {
    console.error("❌ getActive failed:", err);
    return [];
  }
}

/**
 * Add an item to active flips
 */
export async function addActive(item_id) {
  try {
    const res = await fetch(`${API_BASE}/flips/add/${item_id}`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("❌ addActive failed:", err);
    return null;
  }
}

/**
 * Remove an item from active flips
 */
export async function removeActive(item_id) {
  try {
    const res = await fetch(`${API_BASE}/flips/remove/${item_id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("❌ removeActive failed:", err);
    return null;
  }
}

/**
 * Fetch SELL signals
 */
export async function getSellSignals() {
  try {
    const res = await fetch(`${API_BASE}/flips/sell-signals`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return json.data || [];
  } catch (err) {
    console.error("❌ getSellSignals failed:", err);
    return [];
  }
}

/**
 * Close (sell) an active flip
 */
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
