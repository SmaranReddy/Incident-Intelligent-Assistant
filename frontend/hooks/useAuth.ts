"use client";

import { useState, useCallback, useEffect } from "react";
import { login as apiLogin } from "@/lib/api";

export function useAuth() {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("rag_token");
    if (stored) setToken(stored);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiLogin(email, password);
      localStorage.setItem("rag_token", res.access_token);
      setToken(res.access_token);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("rag_token");
    setToken(null);
  }, []);

  return { token, login, logout, loading, error };
}
