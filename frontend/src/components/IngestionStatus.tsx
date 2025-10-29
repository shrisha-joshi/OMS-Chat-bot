'use client';

import React, { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';

interface IngestionLog {
  step: string;
  status: 'PROCESSING' | 'SUCCESS' | 'FAILED';
  message: string;
  metadata?: any;
  timestamp: string;
}

interface IngestionStatusProps {
  docId: string;
  onComplete?: () => void;
}

const stageIcons: Record<string, string> = {
  EXTRACT: '📄',
  CHUNK: '✂️',
  EMBED: '🧠',
  STORE_CHUNKS: '💾',
  INDEX_VECTORS: '🔍',
  EXTRACT_ENTITIES: '🔗',
  JSON_PARSE: '📋',
  JSON_QA: '❓',
};

const stageNames: Record<string, string> = {
  EXTRACT: 'Extracting Text',
  CHUNK: 'Creating Chunks',
  EMBED: 'Generating Embeddings',
  STORE_CHUNKS: 'Storing in Database',
  INDEX_VECTORS: 'Indexing Vectors',
  EXTRACT_ENTITIES: 'Building Knowledge Graph',
  JSON_PARSE: 'Parsing JSON',
  JSON_QA: 'Generating Q&A',
};

export default function IngestionStatus({ docId, onComplete }: IngestionStatusProps) {
  const [logs, setLogs] = useState<IngestionLog[]>([]);
  const [isProcessing, setIsProcessing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Fetch initial status
    const fetchInitialStatus = async () => {
      try {
        const response = await apiClient.get(`/monitoring/documents/${docId}/ingestion-status`) as any;
        if (response.data?.logs) {
          setLogs(response.data.logs);
        }
        if (response.data?.ingest_status === 'SUCCESS' || response.data?.ingest_status === 'FAILED') {
          setIsProcessing(false);
          if (response.data?.ingest_status === 'SUCCESS' && onComplete) {
            onComplete();
          }
        }
      } catch (err) {
        console.error('Failed to fetch initial status:', err);
      }
    };

    fetchInitialStatus();

    // Connect to WebSocket for real-time updates
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/monitoring/ws/ingestion/${docId}`;
    
    try {
      const websocket = new WebSocket(wsUrl);

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'ingestion_progress') {
            setLogs((prev) => {
              const exists = prev.find((log) => log.step === data.step && log.timestamp === data.timestamp);
              if (exists) return prev;
              return [
                ...prev,
                {
                  step: data.step,
                  status: data.status,
                  message: data.message,
                  metadata: data.metadata,
                  timestamp: data.timestamp,
                },
              ];
            });
          } else if (data.type === 'ingestion_complete') {
            setIsProcessing(false);
            if (data.status === 'SUCCESS' && onComplete) {
              onComplete();
            } else if (data.status === 'FAILED') {
              setError('Document ingestion failed');
            }
          }
        } catch (parseErr) {
          console.error('Failed to parse WebSocket message:', parseErr);
        }
      };

      websocket.onerror = () => {
        setError('WebSocket connection error');
      };

      setWs(websocket);

      return () => {
        websocket.close();
      };
    } catch (wsErr) {
      console.error('Failed to connect WebSocket:', wsErr);
      setError('Failed to connect to real-time updates');
    }
  }, [docId, onComplete]);

  const getStageStatus = (step: string) => {
    const log = logs.find((l) => l.step === step);
    if (!log) return 'pending';
    if (log.status === 'SUCCESS') return 'success';
    if (log.status === 'FAILED') return 'failed';
    return 'processing';
  };

  const stages = ['EXTRACT', 'CHUNK', 'EMBED', 'STORE_CHUNKS', 'INDEX_VECTORS', 'EXTRACT_ENTITIES'];

  return (
    <div className="w-full bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg p-6 border border-slate-200">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-2">📊 RAG Pipeline Processing</h3>
        <p className="text-sm text-slate-600">Real-time document ingestion progress</p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          ❌ {error}
        </div>
      )}

      <div className="space-y-3">
        {stages.map((stage, index) => {
          const status = getStageStatus(stage);
          const log = logs.find((l) => l.step === stage);
          const icon = stageIcons[stage] || '⚙️';
          const name = stageNames[stage] || stage;

          return (
            <div key={stage} className="flex items-start gap-3">
              <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center mt-0.5">
                {status === 'success' && <span className="text-green-600 text-lg">✅</span>}
                {status === 'failed' && <span className="text-red-600 text-lg">❌</span>}
                {status === 'processing' && <span className="text-blue-600 text-lg animate-spin">⏳</span>}
                {status === 'pending' && <span className="text-slate-400 text-lg">⏸️</span>}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xl">{icon}</span>
                  <span className={`font-medium ${
                    status === 'success' ? 'text-green-700' :
                    status === 'failed' ? 'text-red-700' :
                    status === 'processing' ? 'text-blue-700' :
                    'text-slate-500'
                  }`}>
                    {name}
                  </span>
                </div>

                {log && (
                  <div className="mt-1 ml-8">
                    <p className="text-sm text-slate-700">{log.message}</p>
                    {log.metadata && Object.keys(log.metadata).length > 0 && (
                      <p className="text-xs text-slate-500 mt-1">
                        {Object.entries(log.metadata)
                          .map(([key, value]) => `${key}: ${value}`)
                          .join(' • ')}
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="flex-shrink-0">
                {log && (
                  <span className="text-xs text-slate-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {isProcessing && (
        <div className="mt-6 pt-4 border-t border-slate-200">
          <div className="inline-flex items-center gap-2 text-sm text-blue-700">
            <span className="animate-pulse">🔄</span>
            <span>Processing document... Please wait</span>
          </div>
        </div>
      )}

      {!isProcessing && !error && (
        <div className="mt-6 pt-4 border-t border-slate-200">
          <div className="inline-flex items-center gap-2 text-sm text-green-700 font-medium">
            <span>✅</span>
            <span>Document ready for querying!</span>
          </div>
        </div>
      )}
    </div>
  );
}
