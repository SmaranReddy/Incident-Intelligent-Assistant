import type {
  AuthResponse,
  KnowledgeBase,
  QueryResponse,
  Document,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("rag_token");
}

function authHeaders(): HeadersInit {
  const token = getToken();
  return token
    ? { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }
    : { "Content-Type": "application/json" };
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg =
      body?.detail ||
      body?.message ||
      `Request failed with status ${res.status}`;
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (res.status === 401) throw new Error("Invalid credentials");
  if (!res.ok) throw new Error("Server not reachable");
  return handleResponse<AuthResponse>(res);
}

// ── Knowledge Bases ───────────────────────────────────────────────────────────

export async function listKBs(): Promise<KnowledgeBase[]> {
  const res = await fetch(`${BASE}/kb`, { headers: authHeaders() });
  return handleResponse<KnowledgeBase[]>(res);
}

export async function createKB(
  name: string,
  description?: string
): Promise<KnowledgeBase> {
  const res = await fetch(`${BASE}/kb`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ name, description }),
  });
  return handleResponse<KnowledgeBase>(res);
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function listDocuments(kbId: string): Promise<Document[]> {
  const res = await fetch(`${BASE}/kb/${kbId}/documents`, { headers: authHeaders() });
  const data = await handleResponse<{ documents: Document[]; total: number }>(res);
  return data.documents;
}

export async function uploadDocument(
  kbId: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<Document> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/kb/${kbId}/documents`);
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const body = JSON.parse(xhr.responseText);
          reject(new Error(body?.detail || `Upload failed (${xhr.status})`));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(formData);
  });
}

// ── Query (non-streaming) ─────────────────────────────────────────────────────

export async function queryKB(
  kbId: string,
  query: string,
  docIds?: string[],
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/kb/${kbId}/query`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ query, ...(docIds?.length ? { doc_ids: docIds } : {}) }),
  });
  if (res.status === 422) {
    throw new Error(
      "No documents indexed in this knowledge base yet. Upload documents first."
    );
  }
  return handleResponse<QueryResponse>(res);
}

// ── Delete document ───────────────────────────────────────────────────────────

export async function deleteDocument(kbId: string, docId: string): Promise<void> {
  const res = await fetch(`${BASE}/kb/${kbId}/documents/${docId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok && res.status !== 204) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail || `Delete failed (${res.status})`);
  }
}

// ── Streaming query ───────────────────────────────────────────────────────────

export async function* streamQuery(
  kbId: string,
  query: string,
  docIds?: string[],
): AsyncGenerator<{ token?: string; done?: boolean; sources?: QueryResponse["sources"]; confidence?: number; cache_hit?: boolean }> {
  const token = getToken();
  const res = await fetch(`${BASE}/kb/${kbId}/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ query, ...(docIds?.length ? { doc_ids: docIds } : {}) }),
  });

  if (res.status === 422) {
    throw new Error(
      "No documents indexed in this knowledge base yet. Upload documents first."
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail || `Stream failed (${res.status})`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;
      const raw = trimmed.slice(5).trim();
      if (raw === "[DONE]") {
        yield { done: true };
        return;
      }
      try {
        yield JSON.parse(raw);
      } catch {
        // ignore malformed SSE lines
      }
    }
  }
}
