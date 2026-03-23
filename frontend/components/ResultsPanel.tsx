"use client";

import { PredictionResult } from "@/services/api";

interface ResultsPanelProps {
  result: PredictionResult;
}

const CLASS_COLORS: Record<string, string> = {
  Eumycetoma: "text-red-400",
  Actinomycetoma: "text-yellow-400",
  Normal: "text-green-400",
};

export default function ResultsPanel({ result }: ResultsPanelProps) {
  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">
          Classification
        </h3>

        <div className="flex items-baseline gap-3 mb-4">
          <span className={`text-3xl font-bold ${CLASS_COLORS[result.class_name] || "text-white"}`}>
            {result.class_name}
          </span>
          <span className="text-lg text-gray-400">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>

        <div className="space-y-2">
          {Object.entries(result.probabilities).map(([name, prob]) => (
            <div key={name} className="flex items-center gap-3">
              <span className="text-sm text-gray-400 w-32 truncate">{name}</span>
              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-teal-500 rounded-full transition-all duration-500"
                  style={{ width: `${(prob as number) * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-14 text-right">
                {((prob as number) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Subtype</p>
          <p className="text-sm font-medium text-gray-200">{result.subtype}</p>
        </div>
        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Bounding Box</p>
          <p className="text-sm font-mono text-gray-300">
            [{result.bounding_box.map((v) => v.toFixed(2)).join(", ")}]
          </p>
        </div>
      </div>
    </div>
  );
}
