'use client'

import React, { useEffect, useState } from 'react'
import { apiClient } from '@/lib/api'

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  service: string
  embedding_model?: string
  llm_endpoint?: string
  features?: {
    vector_search: boolean
    graph_search: boolean
    reranking: boolean
    streaming: boolean
  }
  error?: string
}

export default function SystemStatus() {
  const [status, setStatus] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const checkHealth = async () => {
    try {
      const data = await apiClient.get<HealthStatus>('/chat/health')
      setStatus(data)
      setError(null)
    } catch {
      setStatus(null)
      setError('System Offline')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Poll every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="text-xs text-slate-500">Checking system...</div>

  const isHealthy = status?.status === 'healthy'
  const isDegraded = status?.status === 'degraded'
  
  let colorClass = 'bg-red-500'
  if (isHealthy) colorClass = 'bg-green-500'
  else if (isDegraded) colorClass = 'bg-yellow-500'

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 rounded-full border border-slate-200 group relative cursor-help">
      <div className={`w-2 h-2 rounded-full ${colorClass} animate-pulse`} />
      <span className="text-xs font-medium text-slate-600">
        {error || (isHealthy ? 'System Operational' : 'System Degraded')}
      </span>

      {/* Tooltip */}
      {status && (
        <div className="absolute top-full right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-slate-200 p-3 hidden group-hover:block z-50">
          <h5 className="text-xs font-bold text-slate-700 mb-2 border-b pb-1">System Health</h5>
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Status:</span>
              <span className={`font-medium ${isHealthy ? 'text-green-600' : 'text-yellow-600'}`}>
                {status.status.toUpperCase()}
              </span>
            </div>
            {status.embedding_model && (
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Embeddings:</span>
                <span className="text-slate-700 truncate max-w-[120px]" title={status.embedding_model}>
                  {status.embedding_model}
                </span>
              </div>
            )}
            {status.features && (
              <div className="pt-1 border-t border-slate-100 mt-1">
                <p className="text-[10px] font-semibold text-slate-500 mb-1">ACTIVE FEATURES</p>
                <div className="grid grid-cols-2 gap-1">
                  {Object.entries(status.features).map(([key, active]) => (
                    <div key={key} className="flex items-center gap-1">
                      <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-green-400' : 'bg-slate-300'}`} />
                      <span className="text-[10px] text-slate-600 capitalize">{key.replace('_', ' ')}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
