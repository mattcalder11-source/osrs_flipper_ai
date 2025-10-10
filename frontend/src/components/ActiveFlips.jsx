import React from "react";
import { closeFlip } from "../api";

export default function ActiveFlips({ flips, onRefresh }) {
  if (!flips?.length) return <p className="text-gray-400">No active flips.</p>;

  async function handleSell(item_id) {
    await closeFlip(item_id);
    onRefresh();
  }

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-2xl shadow p-4 mt-4">
      <h2 className="text-xl font-semibold mb-2 text-blue-400">Active Flips</h2>
      <table className="min-w-full text-sm text-center">
        <thead>
          <tr className="border-b border-gray-700 text-gray-300">
            <th className="py-2 px-2 text-left">Item</th>
            <th>Buy Price</th>
            <th>Current Price</th>
            <th>Profit %</th>
            <th>Profit (GP)</th>
            <th>Hold (hrs)</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {flips.map((f) => (
            <tr key={f.item_id} className="border-b border-gray-800 hover:bg-gray-800/50">
              <td className="py-2 px-2 flex items-center gap-2 text-left">
                {f.icon_url && (
                  <img
                    src={f.icon_url}
                    alt={f.name}
                    className="w-5 h-5 inline-block rounded"
                  />
                )}
                {f.name || f.item_id}
              </td>
              <td>{f.entry_price?.toLocaleString("en-US") ?? "—"} gp</td>
              <td>{f.current_price?.toLocaleString("en-US") ?? "—"} gp</td>
              <td
                className={
                  f.profit_pct > 0
                    ? "text-green-400"
                    : f.profit_pct < 0
                    ? "text-red-400"
                    : "text-gray-400"
                }
              >
                {f.profit_pct ? f.profit_pct.toFixed(2) + "%" : "—"}
              </td>
              <td
                className={
                  f.profit_gp > 0
                    ? "text-green-400"
                    : f.profit_gp < 0
                    ? "text-red-400"
                    : "text-gray-400"
                }
              >
                {f.profit_gp?.toLocaleString("en-US") ?? "—"}
              </td>
              <td>{f.hold_hours?.toFixed(1) ?? "—"}</td>
              <td>
                <button
                  onClick={() => handleSell(f.item_id)}
                  className="text-red-400 hover:text-red-200"
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
