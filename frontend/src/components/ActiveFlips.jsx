// src/components/ActiveFlips.jsx
import React, { useEffect, useState } from "react";
import { getActive, closeFlip } from "../api";
import Loading from "./Loading";

export default function ActiveFlips() {
  const [flips, setFlips] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch on mount
  useEffect(() => {
    loadFlips();
    const interval = setInterval(loadFlips, 60000); // refresh every 60s
    return () => clearInterval(interval);
  }, []);

  async function loadFlips() {
    setLoading(true);
    const data = await getActive();
    setFlips(data);
    setLoading(false);
  }

  async function handleSell(itemId) {
    setRefreshing(true);
    await closeFlip(itemId);
    await loadFlips(); // 🔁 auto-refresh after selling
    setRefreshing(false);
  }

  if (loading) return <Loading text="Loading active flips..." />;

  if (!flips.length)
    return (
      <div className="text-center text-gray-400 mt-8">
        No active flips currently being tracked.
      </div>
    );

  return (
    <div className="p-4">
      <h2 className="text-lg font-bold mb-3 flex justify-between items-center">
        Active Flips
        {refreshing && (
          <span className="text-sm text-blue-400 animate-pulse">Refreshing…</span>
        )}
      </h2>
      <table className="min-w-full border border-gray-700 rounded-md text-sm">
        <thead className="bg-gray-800 text-gray-300">
          <tr>
            <th className="px-3 py-2 text-left">Item</th>
            <th className="px-3 py-2">Buy Price</th>
            <th className="px-3 py-2">Current</th>
            <th className="px-3 py-2">Profit (GP)</th>
            <th className="px-3 py-2">%</th>
            <th className="px-3 py-2"></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700">
          {flips.map((flip) => (
            <tr key={flip.item_id} className="hover:bg-gray-800">
              <td className="px-3 py-2 flex items-center space-x-2">
                <img
                  src={flip.icon_url}
                  alt={flip.name}
                  className="w-6 h-6 rounded"
                />
                <span>{flip.name}</span>
              </td>
              <td className="px-3 py-2 text-right">{flip.entry_price?.toLocaleString()}</td>
              <td className="px-3 py-2 text-right">{flip.current_price?.toLocaleString()}</td>
              <td className="px-3 py-2 text-right">
                {flip.profit_gp?.toLocaleString() ?? "-"}
              </td>
              <td className="px-3 py-2 text-right">
                {flip.profit_pct ? flip.profit_pct.toFixed(1) + "%" : "-"}
              </td>
              <td className="px-3 py-2 text-right">
                <button
                  onClick={() => handleSell(flip.item_id)}
                  className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-white text-xs"
                >
                  Sell
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
