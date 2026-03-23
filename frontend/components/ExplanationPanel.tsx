"use client";

interface ExplanationPanelProps {
  explanation: string;
  cached: boolean;
  isLoading: boolean;
}

export default function ExplanationPanel({ explanation, cached, isLoading }: ExplanationPanelProps) {
  if (isLoading) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">
          Clinical Explanation
        </h3>
        <div className="space-y-3">
          <div className="h-4 bg-gray-700 rounded animate-pulse w-full" />
          <div className="h-4 bg-gray-700 rounded animate-pulse w-5/6" />
          <div className="h-4 bg-gray-700 rounded animate-pulse w-4/6" />
        </div>
      </div>
    );
  }

  if (!explanation) return null;

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
          Clinical Explanation
        </h3>
        {cached && (
          <span className="text-xs bg-gray-700 text-gray-400 px-2 py-0.5 rounded">
            cached
          </span>
        )}
      </div>
      <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">{explanation}</p>
    </div>
  );
}
