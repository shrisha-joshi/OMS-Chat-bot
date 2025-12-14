'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { 
  Loader, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Zap, 
  Database, 
  Brain, 
  Search,
  FileText,
  Network,
  Activity
} from 'lucide-react'

interface ProcessingStep {
  id: string
  name: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  progress: number
  duration?: number
  error?: string
}

interface ProcessingIndicatorProps {
  isVisible: boolean
  steps?: ProcessingStep[]
  currentQuery?: string
  onComplete?: () => void
}

export function ProcessingIndicator({ 
  isVisible, 
  steps, 
  currentQuery,
  onComplete 
}: ProcessingIndicatorProps) {
  const [currentSteps, setCurrentSteps] = useState<ProcessingStep[]>([])
  const [totalProgress, setTotalProgress] = useState(0)
  const [startTime, setStartTime] = useState<Date | null>(null)
  const [elapsedTime, setElapsedTime] = useState(0)

  const simulateProcessing = useCallback(async () => {
    const stepOrder = ['parse', 'embed', 'search', 'retrieve', 'rerank', 'generate']
    
    for (const stepId of stepOrder) {
      // Start step
      setCurrentSteps((prev: ProcessingStep[]) => prev.map((step: ProcessingStep) => 
        step.id === stepId 
          ? { ...step, status: 'processing' as const, progress: 0 }
          : step
      ))

      // Simulate progress
      await new Promise(resolve => {
        const progressInterval = setInterval(() => {
          setCurrentSteps((prev: ProcessingStep[]) => {
            const updatedSteps = prev.map((step: ProcessingStep) => {
              if (step.id === stepId && step.status === 'processing') {
                const newProgress = Math.min(step.progress + Math.random() * 25, 100)
                if (newProgress >= 100) {
                  clearInterval(progressInterval)
                  setTimeout(resolve as () => void, 100)
                  return { ...step, status: 'completed' as const, progress: 100, duration: Math.random() * 2 + 0.5 }
                }
                return { ...step, progress: newProgress }
              }
              return step
            })
            
            // Calculate total progress
            const totalProg = updatedSteps.reduce((sum: number, step: ProcessingStep) => sum + step.progress, 0) / updatedSteps.length
            setTotalProgress(totalProg)
            
            return updatedSteps
          })
        }, 50 + Math.random() * 100)
      })

      // Small delay between steps
      await new Promise(resolve => setTimeout(resolve, 200))
    }

    // Complete
    setTimeout(() => {
      onComplete?.()
    }, 500)
  }, [onComplete])

  useEffect(() => {
    if (isVisible) {
      const defaultSteps: ProcessingStep[] = [
        { id: 'parse', name: 'Parsing query', status: 'pending', progress: 0 },
        { id: 'embed', name: 'Generating embeddings', status: 'pending', progress: 0 },
        { id: 'search', name: 'Searching knowledge base', status: 'pending', progress: 0 },
        { id: 'retrieve', name: 'Retrieving documents', status: 'pending', progress: 0 },
        { id: 'rerank', name: 'Reranking results', status: 'pending', progress: 0 },
        { id: 'generate', name: 'Generating response', status: 'pending', progress: 0 }
      ]
      setCurrentSteps(steps || defaultSteps)
      setStartTime(new Date())
      setElapsedTime(0)
      simulateProcessing()
    }
  }, [isVisible, steps, simulateProcessing])

  useEffect(() => {
    if (!isVisible || !startTime) return

    const interval = setInterval(() => {
      setElapsedTime((Date.now() - startTime.getTime()) / 1000)
    }, 100)

    return () => clearInterval(interval)
  }, [isVisible, startTime])

  const getStepIcon = (stepId: string, status: string) => {
    if (status === 'completed') return <CheckCircle className="w-4 h-4 text-green-500" />
    if (status === 'error') return <AlertCircle className="w-4 h-4 text-red-500" />
    if (status === 'processing') return <Loader className="w-4 h-4 text-blue-500 animate-spin" />

    switch (stepId) {
      case 'parse': return <FileText className="w-4 h-4 text-gray-400" />
      case 'embed': return <Brain className="w-4 h-4 text-gray-400" />
      case 'search': return <Search className="w-4 h-4 text-gray-400" />
      case 'retrieve': return <Database className="w-4 h-4 text-gray-400" />
      case 'rerank': return <Activity className="w-4 h-4 text-gray-400" />
      case 'generate': return <Zap className="w-4 h-4 text-gray-400" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const formatTime = (seconds: number) => {
    return `${seconds.toFixed(1)}s`
  }

  if (!isVisible) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-2xl p-6 max-w-md w-full mx-4">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <Network className="w-8 h-8 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            Processing Your Query
          </h3>
          {currentQuery && (
            <p className="text-sm text-gray-600 line-clamp-2">
              &quot;{currentQuery}&quot;
            </p>
          )}
        </div>

        {/* Overall Progress */}
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>Overall Progress</span>
            <span>{Math.round(totalProgress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${totalProgress}%` }}
            />
          </div>
        </div>

        {/* Processing Steps */}
        <div className="space-y-3 mb-6">
          {currentSteps.map((step) => (
            <div key={step.id} className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                {getStepIcon(step.id, step.status)}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${
                    step.status === 'completed' ? 'text-green-700' :
                    step.status === 'processing' ? 'text-blue-700' :
                    step.status === 'error' ? 'text-red-700' :
                    'text-gray-600'
                  }`}>
                    {step.name}
                  </span>
                  
                  {step.duration && (
                    <span className="text-xs text-gray-500">
                      {formatTime(step.duration)}
                    </span>
                  )}
                </div>
                
                {step.status === 'processing' && (
                  <div className="mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-1">
                      <div
                        className="bg-blue-500 h-1 rounded-full transition-all duration-150"
                        style={{ width: `${step.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {step.error && (
                  <p className="text-xs text-red-600 mt-1">{step.error}</p>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Stats */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {formatTime(elapsedTime)}
              </div>
              <div className="text-xs text-gray-500">Elapsed Time</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {currentSteps.filter(s => s.status === 'completed').length}/{currentSteps.length}
              </div>
              <div className="text-xs text-gray-500">Steps Complete</div>
            </div>
          </div>
        </div>

        {/* Technical Details (Expandable) */}
        <details className="mt-4">
          <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
            Technical Details
          </summary>
          <div className="mt-2 p-3 bg-gray-50 rounded text-xs text-gray-600">
            <div className="space-y-1">
              <div>• Vector embeddings: sentence-transformers</div>
              <div>• Knowledge base: Qdrant + ArangoDB</div>
              <div>• Language model: Mistral-3B via LMStudio</div>
              <div>• Processing pipeline: RAG with reranking</div>
            </div>
          </div>
        </details>
      </div>
    </div>
  )
}

// Mini version for inline use
interface MiniProcessingIndicatorProps {
  isProcessing: boolean
  currentStep?: string
  className?: string
}

export function MiniProcessingIndicator({ 
  isProcessing, 
  currentStep = 'Processing...', 
  className = '' 
}: MiniProcessingIndicatorProps) {
  if (!isProcessing) return null

  return (
    <div className={`flex items-center space-x-2 text-sm text-gray-600 ${className}`}>
      <Loader className="w-4 h-4 animate-spin text-blue-500" />
      <span>{currentStep}</span>
      <div className="flex space-x-1">
        <div className="w-1 h-1 bg-blue-500 rounded-full animate-bounce" />
        <div className="w-1 h-1 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
        <div className="w-1 h-1 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
      </div>
    </div>
  )
}