import { addActive } from "../api";
import React from "react";

export default function BuyTable({ buys, onRefresh }) {
  if (!buys?.length) return <p className="text-gray-400">No buy recommendations yet.</p>;

  async function handleImplement(itemId) {
    setImplementing(true);
    await addActive(itemId);
    await refreshActive(); // 🔁 trigger refresh on Active Flips table
    setImplementing(false);
}

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-2xl shadow p-4 mt-4">
      <h2 className="text-xl font-semibold mb-2 text-green-400">Buy Recommendations</h2>
      <table className="min-w-full text-sm text-center">
        <thead>
          <tr className="border-b border-gray-700 text-gray-300">
            <th className="py-2 px-2 text-left">Item</th>
            <th>Buy</th>
            <th>Sell</th>
            <th>Profit (GP)</th>
            <th>Profit %</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {buys.slice(0, 20).map((b) => (
            <tr key={b.item_id} className="border-b border-gray-800 hover:bg-gray-800/50">
              <td className="py-2 px-2 flex items-center gap-2 text-left">
                {b.icon_url && (
                  <img
                    src={b.icon_url}
                    alt={b.name}
                    className="w-5 h-5 inline-block rounded"
                  />
                )}
                {b.name || b.item_id}
              </td>
              <td>{b.buy_price?.toLocaleString("en-US") ?? "—"} gp</td>
              <td>{b.sell_price?.toLocaleString("en-US") ?? "—"} gp</td>
              <td className="text-green-400">
                {b.predicted_profit_gp?.toLocaleString("en-US") ?? "0"}
              </td>
              <td className="text-green-400">
                {b.profit_pct ? b.profit_pct.toFixed(2) + "%" : "—"}
              </td>
              <td>
                <button
                  onClick={() => handleBuy(b.item_id)}
                  className="text-green-400 hover:text-green-200"
                >
                  Implement
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
