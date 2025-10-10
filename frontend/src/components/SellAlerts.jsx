export default function SellAlerts({ signals }) {
  if (!signals?.length) return <p className="text-gray-400">No sell alerts.</p>;

  return (
    <div className="overflow-x-auto bg-gray-900 rounded-2xl shadow p-4 mt-4">
      <h2 className="text-xl font-semibold mb-2 text-red-400">Sell Signals</h2>
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-300">
            <th className="text-left py-2 px-2">Item</th>
            <th>Reason</th>
            <th>Confidence</th>
            <th>Urgency</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr
              key={s.item_id}
              className={`border-b border-gray-800 ${
                s.urgency_score > 0.8
                  ? "bg-red-950"
                  : s.urgency_score > 0.6
                  ? "bg-yellow-900"
                  : ""
              }`}
            >
              <td className="py-2 px-2 flex items-center gap-2">
                {s.icon_url && (
                  <img
                    src={s.icon_url}
                    alt={s.name}
                    className="w-5 h-5 inline-block rounded"
                  />
                )}
                {s.name || s.item_id}
              </td>
              <td>{s.reason || "—"}</td>
              <td>{((s.confidence || 0) * 100).toFixed(0)}%</td>
              <td>{((s.urgency_score || 0) * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
