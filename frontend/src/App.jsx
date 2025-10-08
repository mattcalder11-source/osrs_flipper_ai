import { useEffect, useState } from "react";
import BuyTable from "./components/BuyTable";
import ActiveFlips from "./components/ActiveFlips";
import SellAlerts from "./components/SellAlerts";
import HistoryTable from "./components/HistoryTable";
import { getBuys, getSellSignals } from "./api";
import React from "react";

export default function App() {
  const [buys, setBuys] = useState([]);
  const [active, setActive] = useState([]);
  const [signals, setSignals] = useState([]);
  const [history, setHistory] = useState([]);

  // Load local persistence
  useEffect(() => {
    const a = localStorage.getItem("activeFlips");
    const h = localStorage.getItem("flipHistory");
    if (a) setActive(JSON.parse(a));
    if (h) setHistory(JSON.parse(h));
  }, []);

  // Persist changes
  useEffect(() => {
    localStorage.setItem("activeFlips", JSON.stringify(active));
  }, [active]);

  useEffect(() => {
    localStorage.setItem("flipHistory", JSON.stringify(history));
  }, [history]);

  async function refreshData() {
    try {
      const [b, s] = await Promise.all([getBuys(), getSellSignals()]);
      setBuys(b);
      setSignals(s);
    } catch (err) {
      console.error("Fetch error:", err);
    }
  }

  useEffect(() => {
    refreshData();
  }, []);

  // Handle marking a buy as implemented
  function handleImplement(buy) {
    if (active.find((a) => a.item_id === buy.item_id)) return;
    const newFlip = {
      ...buy,
      entry_price: buy.low,
      timestamp: new Date().toISOString(),
    };
    setActive([...active, newFlip]);
  }

  // Handle selling an active flip
  function handleSell(item_id, sell_price) {
    const flip = active.find((a) => a.item_id === item_id);
    if (!flip) return;

    const profit = sell_price - flip.entry_price;
    const profit_pct = (profit / flip.entry_price) * 100;

    const record = {
      item_id,
      name: flip.name,
      entry_price: flip.entry_price,
      sell_price,
      profit,
      profit_pct,
      closed_at: new Date().toISOString(),
    };

    // update state
    setActive(active.filter((a) => a.item_id !== item_id));
    setHistory([record, ...history]); // prepend newest
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 space-y-8">
      <h1 className="text-3xl font-bold mb-6 text-green-400">
        💹 OSRS AI Flipper Dashboard
      </h1>

      <BuyTable buys={buys} onImplement={handleImplement} />
      <ActiveFlips active={active} onSell={handleSell} />
      <SellAlerts signals={signals} />
      <HistoryTable history={history} />
    </div>
  );
}
