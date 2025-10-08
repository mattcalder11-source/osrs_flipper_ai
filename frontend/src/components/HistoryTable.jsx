import React from "react";

export default function HistoryTable({ history }) {
  if (!history?.length) return <p className="text-gray-400">No completed flips yet.</p>;

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-2xl shadow p-4 mt-4">
      <h2 className="text-xl font-semibold mb-2 text-blue-400">Flip History</h2>
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-300">
            <th className="text-left py-2 px-2">Item</th>
            <th className="text-right py-2 px-2">Buy Price</th>
            <th className="text-right py-2 px-2">Sell Price</th>
            <th className="text-right py-2 px-2">Profit (GP)</th>
            <th className="text-right py-2 px-2">Profit %</th>
            <th className="text-right py-2 px-2">Closed</th>
          </tr>
        </thead>
        <tbody>
          {history.map((h, idx) => (
            <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/60">
              <td className="py-2 px-2">{h.name}</td>
              <td className="text-right py-2 px-2">{h.entry_price.toLocaleString()}</td>
              <td className="text-right py-2 px-2">{h.sell_price.toLocaleString()}</td>
              <td
                className={`text-right py-2 px-2 ${
                  h.profit >= 0 ? "text-green-400" : "text-red-400"
                }`}
              >
                {h.profit.toFixed(0)}
              </td>
              <td
                className={`text-right py-2 px-2 ${
                  h.profit_pct >= 0 ? "text-green-400" : "text-red-400"
                }`}
              >
                {h.profit_pct.toFixed(2)}%
              </td>
              <td className="text-right py-2 px-2 text-gray-400">
                {new Date(h.closed_at).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
