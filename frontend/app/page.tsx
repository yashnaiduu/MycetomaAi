"use client";

import { useState } from "react";
import ImageUpload from "@/components/ImageUpload";
import ResultsPanel from "@/components/ResultsPanel";
import HeatmapViewer from "@/components/HeatmapViewer";
import ExplanationPanel from "@/components/ExplanationPanel";
import {
  predictImage,
  getExplanation,
  PredictionResult,
} from "@/services/api";

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [explanation, setExplanation] = useState("");
  const [explainCached, setExplainCached] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async (file: File) => {
    setError("");
    setPrediction(null);
    setExplanation("");
    setIsAnalyzing(true);

    try {
      const result = await predictImage(file);
      setPrediction(result);

      setIsExplaining(true);
      const explainResult = await getExplanation(result);
      setExplanation(explainResult.explanation);
      setExplainCached(explainResult.cached);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setError(msg);
    } finally {
      setIsAnalyzing(false);
      setIsExplaining(false);
    }
  };

  return (
    <main className="max-w-6xl mx-auto px-6 py-12">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          Mycetoma AI <span className="text-teal-400">Diagnostics</span>
        </h1>
        <p className="text-gray-400 max-w-lg mx-auto">
          Upload a histopathology image for AI-powered classification,
          grain localization, and clinical explanation.
        </p>
      </header>

      <section className="max-w-xl mx-auto mb-12">
        <ImageUpload onFileSelected={handleAnalyze} isLoading={isAnalyzing} />
      </section>

      {error && (
        <div className="max-w-xl mx-auto mb-8 p-4 bg-red-900/30 border border-red-700 rounded-xl text-red-300 text-sm">
          {error}
        </div>
      )}

      {prediction && (
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <ResultsPanel result={prediction} />
            <ExplanationPanel
              explanation={explanation}
              cached={explainCached}
              isLoading={isExplaining}
            />
          </div>
          <div>
            <HeatmapViewer heatmapBase64={prediction.heatmap_base64} />
          </div>
        </div>
      )}
    </main>
  );
}
