import React, { useState } from "react";

export default function ActiveFlips({ active, onSell }) {
  const [sellInputs, setSellInputs] = useState({});

  if (!active?.length) return <p className="text-gray-400">No active flips.</p>;

  const handleChange = (id, value) => {
    setSellInputs({ ...sellInputs, [id]: value });
  };

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-2xl shadow p-4 mt-4">
      <h2 className="text-xl font-semibold mb-2 text-yellow-400">Active Flips</h2>
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-300">
            <th>Item</th>
            <th>Entry Price</th>
            <th>Sell Price</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {active.map((a) => (
            <tr key={a.item_id} className="border-b border-gray-800">
              <td className="py-2 px-2">{a.name}</td>
              <td className="text-right py-2 px-2">{a.entry_price.toLocaleString()}</td>
              <td className="text-right py-2 px-2">
                <input
                  type="number"
                  placeholder="Sell..."
                  value={sellInputs[a.item_id] || ""}
                  onChange={(e) => handleChange(a.item_id, e.target.value)}
                  className="bg-gray-800 text-gray-100 w-24 rounded px-2 py-1 text-right"
                />
              </td>
              <td className="text-right py-2 px-2">
                <button
                  onClick={() =>
                    onSell(a.item_id, parseFloat(sellInputs[a.item_id] || 0))
                  }
                  className="text-red-400 hover:text-red-200"
                >
                  Sold ✖
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
