"use client";

import { useState, useCallback, useRef } from "react";
import { streamQuery } from "@/lib/api";
import type { Message, Source } from "@/lib/types";

function newId() {
  return Math.random().toString(36).slice(2);
}

export function useChat(kbId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!kbId || isStreaming) return;

      const userMsg: Message = {
        id: newId(),
        role: "user",
        content: query,
        timestamp: new Date(),
      };

      const assistantId = newId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      let cancelled = false;
      abortRef.current = () => { cancelled = true; };

      const startTime = Date.now();

      try {
        const gen = streamQuery(kbId, query);

        for await (const chunk of gen) {
          if (cancelled) break;

          if (chunk.token) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + chunk.token }
                  : m
              )
            );
          }

          if (chunk.done || chunk.sources !== undefined) {
            const latency = Date.now() - startTime;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      isStreaming: false,
                      sources: (chunk.sources as Source[]) ?? m.sources,
                      confidence: chunk.confidence ?? m.confidence,
                      cache_hit: chunk.cache_hit ?? m.cache_hit,
                      latency_ms: latency,
                    }
                  : m
              )
            );
          }
        }
      } catch (e) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : "Something went wrong";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: "", error: msg, isStreaming: false }
                : m
            )
          );
        }
      } finally {
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m
          )
        );
      }
    },
    [kbId, isStreaming]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const stopStreaming = useCallback(() => {
    abortRef.current?.();
    setIsStreaming(false);
  }, []);

  return { messages, isStreaming, sendMessage, clearMessages, stopStreaming };
}
