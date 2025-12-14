'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Array<{
    filename: string
    similarity: number
    text: string
  }>
  attachments?: Array<{
    type: string
    url?: string
  }>
  processing_time?: number
  tokens_generated?: number
}

interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

interface ChatContextType {
  sessions: ChatSession[]
  currentSession: ChatSession | null
  isConnected: boolean
  isProcessing: boolean
  createNewSession: () => ChatSession
  switchSession: (sessionId: string) => void
  sendMessage: (message: string) => void
  clearSessions: () => void
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

const API_BASE_URL = globalThis.window !== undefined
  ? (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000')
  : 'http://localhost:8000'

export function ChatProvider({ children }: Readonly<{ children: ReactNode }>) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [isConnected] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)

  // Load sessions from localStorage
  useEffect(() => {
    if (globalThis.window === undefined) return
    
    try {
      const savedSessions = localStorage.getItem('chat_sessions')
      if (savedSessions) {
        const parsedSessions = JSON.parse(savedSessions)
        setSessions(parsedSessions)
        if (parsedSessions.length > 0) {
          setCurrentSession(parsedSessions[0])
        }
      }
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [])

  // Save sessions to localStorage
  useEffect(() => {
    if (globalThis.window === undefined) return
    
    try {
      localStorage.setItem('chat_sessions', JSON.stringify(sessions))
    } catch (e) {
      console.error('Failed to save sessions:', e)
    }
  }, [sessions])

  const createNewSession = useCallback((): ChatSession => {
    const newSession: ChatSession = {
      id: `session_${Date.now()}`,
      title: 'New Chat',
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    setSessions(prev => [newSession, ...prev])
    setCurrentSession(newSession)
    return newSession
  }, [])

  const switchSession = useCallback((sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId)
    if (session) {
      setCurrentSession(session)
    }
  }, [sessions])

  const sendMessage = useCallback(async (content: string) => {
    if (!currentSession || isProcessing || !content.trim()) return

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }

    let updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
      updated_at: new Date().toISOString(),
      title: currentSession.messages.length === 0 ? content.slice(0, 50) : currentSession.title
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))

    // Add loading message
    const loadingMessage: ChatMessage = {
      id: `msg_${Date.now() + 1}`,
      role: 'assistant',
      content: '🤔 Processing your question... This may take up to 2-3 minutes for complex queries.',
      timestamp: new Date().toISOString()
    }

    updatedSession = {
      ...updatedSession,
      messages: [...updatedSession.messages, loadingMessage]
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
    setIsProcessing(true)

    try {
      // Call REST API with extended timeout for LLM responses (3 minutes)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => {
        console.warn('⏰ Request timeout after 3 minutes, aborting...')
        controller.abort()
      }, 180000) // 3 minutes
      
      console.log('📤 Sending query to backend:', content.substring(0, 100))
      
      const response = await fetch(`${API_BASE_URL}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: content,
          session_id: currentSession.id,
          context: []
        }),
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)

      console.log('✓ Fetch completed, status:', response.status, response.statusText)
      console.log('✓ Response headers:', Object.fromEntries(response.headers.entries()))

      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Backend error response:', errorText)
        throw new Error(`API error: ${response.statusText} - ${errorText}`)
      }

      console.log('✓ Parsing JSON response...')
      const data = await response.json()
      console.log('✓ Response received from backend:', data)
      console.log('✓ Response content:', data.response ? data.response.substring(0, 100) : 'NO RESPONSE')

      // Validate response has content
      if (!data.response && !data.answer && !data.message) {
        console.error('❌ Response has no content field:', Object.keys(data))
        throw new Error('Backend response missing content field')
      }

      // Replace loading message with response
      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: data.response || data.answer || data.message || 'I apologize, but I could not generate a response.',
        timestamp: new Date().toISOString(),
        sources: data.sources || [],
        attachments: data.attachments || [],
        processing_time: data.processing_time,
        tokens_generated: data.tokens_generated
      }

      console.log('✓ Created assistant message with', assistantMessage.content.length, 'characters')

      console.log('✓ Created assistant message with', assistantMessage.content.length, 'characters')

      updatedSession = {
        ...updatedSession,
        messages: [...updatedSession.messages.slice(0, -1), assistantMessage],
        updated_at: new Date().toISOString()
      }

      console.log('✓ Updating session with', updatedSession.messages.length, 'messages')
      setCurrentSession(updatedSession)
      setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
      console.log('✓ Chat response processed successfully!')
    } catch (error) {
      console.error('❌ Failed to send message:', error)
      console.error('Error details:', {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      })

      // Show error message
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}. Please check the browser console (F12) for details.`,
        timestamp: new Date().toISOString()
      }

      updatedSession = {
        ...updatedSession,
        messages: [...updatedSession.messages.slice(0, -1), errorMessage]
      }

      setCurrentSession(updatedSession)
      setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
    } finally {
      setIsProcessing(false)
    }
  }, [currentSession, isProcessing])

  const clearSessions = useCallback(() => {
    setSessions([])
    setCurrentSession(null)
    if (globalThis.window !== undefined) {
      localStorage.removeItem('chat_sessions')
    }
  }, [])

  const value = React.useMemo(() => ({
    sessions,
    currentSession,
    isConnected,
    isProcessing,
    createNewSession,
    switchSession,
    sendMessage,
    clearSessions
  }), [sessions, currentSession, isConnected, isProcessing, createNewSession, switchSession, sendMessage, clearSessions])

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider')
  }
  return context
}