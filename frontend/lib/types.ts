export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  document_count?: number;
}

export interface Source {
  chunk_id: string;
  text: string;
  metadata: Record<string, unknown>;
  score?: number;
  document_name?: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  cache_hit: boolean;
  latency_ms?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  confidence?: number;
  cache_hit?: boolean;
  latency_ms?: number;
  isStreaming?: boolean;
  error?: string;
  timestamp: Date;
}

export interface Document {
  id: string;
  filename: string;
  status: "pending" | "processing" | "indexed" | "error";
  created_at?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

export interface StreamChunk {
  token?: string;
  done?: boolean;
  sources?: Source[];
  confidence?: number;
  cache_hit?: boolean;
  error?: string;
}
