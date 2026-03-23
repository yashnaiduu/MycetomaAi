import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 60000,
});

export interface PredictionResult {
  class_name: string;
  class_id: number;
  confidence: number;
  probabilities: Record<string, number>;
  bounding_box: number[];
  subtype: string;
  subtype_id: number;
  heatmap_base64: string;
}

export interface ExplanationResult {
  explanation: string;
  cached: boolean;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post<PredictionResult>("/predict/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getExplanation(
  prediction: PredictionResult
): Promise<ExplanationResult> {
  const { data } = await api.post<ExplanationResult>("/explain/", {
    class_name: prediction.class_name,
    confidence: prediction.confidence,
    subtype: prediction.subtype,
    probabilities: prediction.probabilities,
  });
  return data;
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}
