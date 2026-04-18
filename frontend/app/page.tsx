"use client";

import { useState } from "react";

interface SimilarIncident {
  incident_id: string;
  service: string;
  score: number;
}

interface QueryResult {
  query: string;
  likely_root_cause: string;
  suggested_fix: string;
  similar_incidents: SimilarIncident[];
  affected_services: string[];
  confidence: number;
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 75 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-red-500";
  const textColor =
    pct >= 75 ? "text-emerald-400" : pct >= 50 ? "text-amber-400" : "text-red-400";

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-sm font-semibold tabular-nums w-10 text-right ${textColor}`}>
        {pct}%
      </span>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? "bg-emerald-900/60 text-emerald-300 border-emerald-700" :
    pct >= 60 ? "bg-amber-900/60 text-amber-300 border-amber-700" :
                "bg-gray-800 text-gray-400 border-gray-600";
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded border ${color}`}>
      {pct}%
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12" cy="12" r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

export default function IncidentDebugger() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }

      const data: QueryResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0d14] text-gray-100 font-sans">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <h1 className="text-base font-semibold text-gray-200 tracking-tight">
            Incident Debugger
          </h1>
          <span className="ml-auto text-xs text-gray-600 font-mono">RAG · v1</span>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 space-y-8">
        {/* Input */}
        <div className="space-y-3">
          <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider">
            Describe the incident or symptom
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              disabled={loading}
              placeholder="e.g. payment timeout after checkout"
              className="flex-1 bg-[#13151f] border border-gray-700 rounded-lg px-4 py-2.5
                         text-sm text-gray-100 placeholder-gray-600 outline-none
                         focus:border-blue-600 focus:ring-1 focus:ring-blue-600/40
                         disabled:opacity-50 transition"
            />
            <button
              onClick={submit}
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 active:bg-blue-800
                         disabled:opacity-40 disabled:cursor-not-allowed
                         rounded-lg px-5 py-2.5 text-sm font-medium transition"
            >
              {loading ? <><Spinner /> Analyzing</> : "Analyze"}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950/50 border border-red-800 rounded-lg px-4 py-3 text-sm text-red-300">
            <span className="font-medium">Error:</span> {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4 animate-fade-in">

            {/* Root Cause */}
            <div className="bg-red-950/30 border border-red-900/60 rounded-lg p-5 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-red-400 uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                Likely Root Cause
              </div>
              <p className="text-sm text-gray-200 leading-relaxed">
                {result.likely_root_cause}
              </p>
            </div>

            {/* Suggested Fix */}
            <div className="bg-emerald-950/30 border border-emerald-900/60 rounded-lg p-5 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Suggested Fix
              </div>
              <p className="text-sm text-gray-200 leading-relaxed">
                {result.suggested_fix}
              </p>
            </div>

            {/* Confidence + Affected Services row */}
            <div className="grid grid-cols-2 gap-4">
              {/* Confidence */}
              <div className="bg-[#13151f] border border-gray-800 rounded-lg p-5 space-y-3">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Confidence
                </p>
                <ConfidenceBar value={result.confidence} />
              </div>

              {/* Affected Services */}
              <div className="bg-[#13151f] border border-gray-800 rounded-lg p-5 space-y-3">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Affected Services
                </p>
                {result.affected_services.length === 0 ? (
                  <p className="text-xs text-gray-600 italic">None identified</p>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {result.affected_services.map((svc) => (
                      <span
                        key={svc}
                        className="text-xs bg-gray-800 border border-gray-700 text-gray-300
                                   rounded px-2 py-0.5 font-mono"
                      >
                        {svc}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Similar Incidents */}
            <div className="bg-[#13151f] border border-gray-800 rounded-lg overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-800 flex items-center gap-2">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Similar Incidents
                </p>
                <span className="ml-auto text-xs text-gray-600">
                  {result.similar_incidents.length} found
                </span>
              </div>

              {result.similar_incidents.length === 0 ? (
                <p className="px-5 py-4 text-xs text-gray-600 italic">
                  No similar incidents found.
                </p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-gray-500 border-b border-gray-800">
                      <th className="px-5 py-2 text-left font-medium">Incident</th>
                      <th className="px-5 py-2 text-left font-medium">Service</th>
                      <th className="px-5 py-2 text-right font-medium">Match</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.similar_incidents.map((inc, i) => (
                      <tr
                        key={inc.incident_id}
                        className={`border-b border-gray-800/50 last:border-0
                          ${i === 0 ? "bg-blue-950/20" : "hover:bg-gray-800/30"} transition`}
                      >
                        <td className="px-5 py-3 font-mono text-blue-400 text-xs">
                          {i === 0 && (
                            <span className="mr-1.5 text-blue-500 text-[10px] font-semibold">
                              TOP
                            </span>
                          )}
                          {inc.incident_id}
                        </td>
                        <td className="px-5 py-3 text-gray-300 text-xs">{inc.service}</td>
                        <td className="px-5 py-3 text-right">
                          <ScoreBadge score={inc.score} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

          </div>
        )}

        {/* Empty state */}
        {!result && !error && !loading && (
          <div className="text-center py-16 text-gray-700 text-sm">
            Enter a query above to analyze an incident.
          </div>
        )}
      </main>
    </div>
  );
}
