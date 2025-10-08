export default function SellAlerts({ signals }) {
  if (!signals?.length) return null;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-2">Sell Signals</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400">
            <th>Item</th><th>Reason</th><th>Confidence</th><th>Urgency</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => (
            <tr key={s.item_id}
              className={`border-b border-gray-800 ${
                s.urgency_score > 0.8
                  ? "bg-red-950"
                  : s.urgency_score > 0.6
                  ? "bg-yellow-900"
                  : ""
              }`}
            >
              <td>{s.item_id}</td>
              <td>{s.reason}</td>
              <td>{(s.confidence * 100).toFixed(0)}%</td>
              <td>{(s.urgency_score * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
