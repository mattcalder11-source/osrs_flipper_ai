import { addActive } from "../api";
import React from "react";


export default function BuyTable({ buys, onRefresh }) {
if (!buys?.length) return <p>No buy recommendations yet.</p>;


async function handleBuy(item_id) {
await addActive(item_id);
onRefresh();
}


return (
<div>
<h2 className="text-xl font-semibold mb-2">Buy Recommendations</h2>
<table className="w-full text-sm text-center">
<thead>
<tr className="text-gray-400">
<th className="text-center">Item</th>
<th className="text-center">Buy</th>
<th className="text-center">Sell</th>
<th className="text-center">Profit %</th>
<th className="text-center">Action</th>
</tr>
</thead>
<tbody>
{buys.slice(0, 20).map((b) => (
<tr key={b.item_id} className="border-b border-gray-800 text-center">
<td>{b.name}</td>
<td>{b.low?.toLocaleString("en-US")} gp</td>
<td>{b.high?.toLocaleString("en-US")} gp</td>
<td>{(b.potential_profit * 100).toFixed(2)}%</td>
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