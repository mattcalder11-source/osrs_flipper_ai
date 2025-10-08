export default function Loading({ text = "Loading..." }) {
  return (
    <div className="flex justify-center items-center py-12">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-400 mr-3"></div>
      <span className="text-gray-300">{text}</span>
    </div>
  );
}
