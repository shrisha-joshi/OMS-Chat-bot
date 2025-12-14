'use client';

import React, { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api';

interface IngestionLog {
  step: string;
  status: 'PROCESSING' | 'SUCCESS' | 'FAILED';
  message: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

export default function IngestionStatus({ docId, onComplete }: Readonly<IngestionStatusProps>) {
  const [logs, setLogs] = useState<IngestionLog[]>([]);
  const [isProcessing, setIsProcessing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Helper: process parsed websocket message
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleIngestionEvent = React.useCallback((data: any) => {
    if (data.type === 'progress') {
      // Map new backend format to component state
      setLogs((prev) => {
        // Check if we already have this update
        const exists = prev.find((log) => log.step === data.stage && log.status === data.status);
        if (exists) return prev;
        
        // Remove any previous log for this stage to update it
        const filtered = prev.filter(log => log.step !== data.stage);
        
        return [
          ...filtered,
          {
            step: data.stage,
            status: data.status,
            message: data.message,
            metadata: {},
            timestamp: new Date().toISOString(),
          },
        ];
      });

      if (data.status === 'FAILED') {
        setError(data.message || 'Document ingestion failed');
        setIsProcessing(false);
      } else if (data.stage === 'COMPLETE' || (data.stage === 'EXTRACT_ENTITIES' && data.status === 'SUCCESS')) {
        // If we hit the last stage or explicit complete
        if (data.stage === 'COMPLETE') {
             setIsProcessing(false);
             if (onComplete) onComplete();
        }
      }
    }
  }, [onComplete]);

  useEffect(() => {
    // Fetch initial status
    const fetchInitialStatus = async () => {
      try {
        // Use the correct admin endpoint
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const response = await apiClient.get(`/admin/documents/status/${docId}`) as any;
        
        if (response.stages) {
          // Map backend stages to frontend logs format
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const mappedLogs = response.stages.map((stage: any) => ({
            step: stage.name,
            status: stage.status === 'PENDING' ? 'pending' : stage.status, // Handle PENDING status
            message: stage.message,
            metadata: stage.metadata,
            timestamp: stage.timestamp || new Date().toISOString()
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          })).filter((l: any) => l.status !== 'PENDING'); // Only show active/completed stages
          
          setLogs(mappedLogs);
        }
        
        if (response.ingest_status === 'completed' || response.ingest_status === 'failed') {
          setIsProcessing(false);
          if (response.ingest_status === 'completed' && onComplete) {
            onComplete();
          }
        }
      } catch (err) {
        console.error('Failed to fetch initial status:', err);
      }
    };

    fetchInitialStatus();

    // Connect to WebSocket for real-time updates
    // Use backend WebSocket URL from environment or default to 127.0.0.1:8000
    const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://127.0.0.1:8000';
    const wsUrl = `${WS_BASE}/ws/document/${docId}`;
    
    let websocket: WebSocket | null = null;
    let retryCount = 0;
    const maxRetries = 3;

    const connectWebSocket = () => {
      try {
        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = () => {
          console.log('✅ WebSocket Connected');
          setError(null);
          retryCount = 0;
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleIngestionEvent(data);
          } catch (error_) {
            console.error('Failed to parse WebSocket message:', error_);
          }
        };

        websocket.onerror = (e) => {
          console.error('WebSocket error:', e);
          // Don't set error immediately, let retry handle it
        };

        websocket.onclose = (e) => {
          console.log(`WebSocket closed (code: ${e.code})`);
          if (!e.wasClean && retryCount < maxRetries && isProcessing) {
            retryCount++;
            const timeout = Math.min(1000 * retryCount, 5000);
            console.log(`🔄 Retrying connection in ${timeout}ms...`);
            setTimeout(connectWebSocket, timeout);
          } else if (!e.wasClean) {
             // Only show error if we exhausted retries
             // setError('Real-time updates disconnected');
          }
        };

        wsRef.current = websocket;
      } catch (error_) {
        console.error('Failed to connect WebSocket:', error_);
        setError('Failed to connect to real-time updates');
      }
    };

    connectWebSocket();

    return () => {
      if (websocket) {
        console.log('🧹 Cleaning up WebSocket');
        websocket.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId, onComplete, handleIngestionEvent]); // Removed isProcessing from deps to avoid re-connection loops

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
        {stages.map((stage) => {
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
                  {(() => {
                    let colorClass = 'text-slate-500';
                    if (status === 'success') colorClass = 'text-green-700';
                    else if (status === 'failed') colorClass = 'text-red-700';
                    else if (status === 'processing') colorClass = 'text-blue-700';
                    return (
                      <span className={`font-medium ${colorClass}`}>
                        {name}
                      </span>
                    );
                  })()}
                </div>

                {log && (
                  <div className="mt-1 ml-8">
                    <p className="text-sm text-slate-700">{log.message}</p>
                    {log.metadata && Object.keys(log.metadata).length > 0 && (
                      <p className="text-xs text-slate-500 mt-1">
                        {Object.entries(log.metadata)
                          .map(([key, value]) => {
                            let val: string;
                            if (value === null || value === undefined) {
                              val = '';
                            } else if (typeof value === 'object') {
                              try {
                                val = JSON.stringify(value);
                              } catch {
                                val = '[object]';
                              }
                            } else if (typeof value === 'string') {
                              val = value;
                            } else if (typeof value === 'number') {
                              val = Number.isFinite(value) ? value.toString() : 'NaN';
                            } else if (typeof value === 'boolean') {
                              val = value ? 'true' : 'false';
                            } else {
                              val = '';
                            }
                            return `${key}: ${val}`;
                          })
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
