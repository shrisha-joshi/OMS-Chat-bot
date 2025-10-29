'use client';

import React from 'react';

interface RAGMetrics {
  retrieval_time_ms?: number;
  embedding_cache_hit?: boolean;
  vector_results?: number;
  bm25_results?: number;
  hyde_results?: number;
  graph_results?: number;
  reranking_time_ms?: number;
  llm_generation_time_ms?: number;
  total_time_ms?: number;
  tokens_generated?: number;
  query_variants?: string[];
  contextual_enhancement?: boolean;
}

interface RAGPipelineStatusProps {
  metrics?: RAGMetrics;
  sources?: Array<{
    document: string;
    chunk: string;
    score: number;
  }>;
}

export default function RAGPipelineStatus({ metrics, sources }: RAGPipelineStatusProps) {
  if (!metrics && !sources) return null;

  return (
    <div className="mt-4 pt-4 border-t border-slate-200 bg-slate-50 rounded-lg p-3">
      {/* Header */}
      <h4 className="text-xs font-semibold text-slate-700 mb-3 flex items-center gap-1">
        <span>🔄</span> RAG Pipeline Analysis
      </h4>

      {/* Metrics Grid */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
          {metrics.vector_results !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">Vector Results</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.vector_results}</p>
            </div>
          )}

          {metrics.bm25_results !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">Keyword Results</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.bm25_results}</p>
            </div>
          )}

          {metrics.graph_results !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">Graph Relations</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.graph_results}</p>
            </div>
          )}

          {metrics.embedding_cache_hit && (
            <div className="bg-green-50 p-2 rounded border border-green-100">
              <p className="text-xs text-green-700">Cache Hit ✅</p>
              <p className="text-sm font-semibold text-green-900">Embedding Reused</p>
            </div>
          )}

          {metrics.retrieval_time_ms !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">Retrieval Time</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.retrieval_time_ms}ms</p>
            </div>
          )}

          {metrics.llm_generation_time_ms !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">LLM Time</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.llm_generation_time_ms}ms</p>
            </div>
          )}

          {metrics.total_time_ms !== undefined && (
            <div className="bg-blue-50 p-2 rounded border border-blue-100">
              <p className="text-xs text-blue-700">Total Time</p>
              <p className="text-sm font-semibold text-blue-900">{metrics.total_time_ms}ms</p>
            </div>
          )}

          {metrics.tokens_generated !== undefined && (
            <div className="bg-white p-2 rounded border border-slate-100">
              <p className="text-xs text-slate-500">Tokens</p>
              <p className="text-sm font-semibold text-slate-900">{metrics.tokens_generated}</p>
            </div>
          )}
        </div>
      )}

      {/* Query Variants */}
      {metrics?.query_variants && metrics.query_variants.length > 1 && (
        <div className="mb-3 text-xs">
          <p className="font-semibold text-slate-700 mb-1">Query Variants (Phase 3):</p>
          <div className="space-y-1">
            {metrics.query_variants.map((variant, idx) => (
              <p key={idx} className="text-slate-600 italic">
                •  {variant}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Sources */}
      {sources && sources.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-700 mb-2">📚 Retrieved Sources ({sources.length}):</p>
          <div className="space-y-2">
            {sources.slice(0, 3).map((source, idx) => (
              <div key={idx} className="bg-white p-2 rounded border border-slate-100 text-xs">
                <div className="flex justify-between items-start mb-1">
                  <span className="font-semibold text-slate-900">{idx + 1}. {source.document}</span>
                  <span className="text-slate-500">Score: {(source.score * 100).toFixed(0)}%</span>
                </div>
                <p className="text-slate-600 line-clamp-2">{source.chunk}</p>
              </div>
            ))}
            {sources.length > 3 && (
              <p className="text-xs text-slate-500 italic">
                ... and {sources.length - 3} more sources
              </p>
            )}
          </div>
        </div>
      )}

      {/* Pipeline Features */}
      {metrics?.contextual_enhancement && (
        <div className="mt-2 pt-2 border-t border-slate-200">
          <p className="text-xs text-slate-600">
            <span className="font-semibold">✨ Contextual Enhancement:</span> Added surrounding context for better relevance
          </p>
        </div>
      )}
    </div>
  );
}
