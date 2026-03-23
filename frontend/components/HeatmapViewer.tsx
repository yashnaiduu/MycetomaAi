"use client";

interface HeatmapViewerProps {
  heatmapBase64: string;
}

export default function HeatmapViewer({ heatmapBase64 }: HeatmapViewerProps) {
  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">
        Grad-CAM Attention Map
      </h3>
      <img
        src={`data:image/png;base64,${heatmapBase64}`}
        alt="Grad-CAM heatmap"
        className="w-full rounded-lg"
      />
      <p className="text-xs text-gray-500 mt-3">
        Highlighted regions indicate areas the model focused on for diagnosis.
      </p>
    </div>
  );
}
