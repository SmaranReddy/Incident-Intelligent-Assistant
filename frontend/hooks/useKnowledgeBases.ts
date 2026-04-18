"use client";

import { useState, useCallback } from "react";
import { listKBs, createKB } from "@/lib/api";
import type { KnowledgeBase } from "@/lib/types";

export function useKnowledgeBases() {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchKBs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listKBs();
      setKbs(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load knowledge bases");
    } finally {
      setLoading(false);
    }
  }, []);

  const addKB = useCallback(
    async (name: string, description?: string): Promise<KnowledgeBase | null> => {
      try {
        const kb = await createKB(name, description);
        setKbs((prev) => [kb, ...prev]);
        return kb;
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to create knowledge base");
        return null;
      }
    },
    []
  );

  return { kbs, loading, error, fetchKBs, addKB };
}
