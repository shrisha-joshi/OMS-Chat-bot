'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

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

const API_BASE_URL = typeof window !== 'undefined' 
  ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  : 'http://localhost:8000'

export function ChatProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [isConnected] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)

  // Load sessions from localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return
    
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
    if (typeof window === 'undefined') return
    
    try {
      localStorage.setItem('chat_sessions', JSON.stringify(sessions))
    } catch (e) {
      console.error('Failed to save sessions:', e)
    }
  }, [sessions])

  const createNewSession = (): ChatSession => {
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
  }

  const switchSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId)
    if (session) {
      setCurrentSession(session)
    }
  }

  const sendMessage = async (content: string) => {
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
      content: '⏳ Thinking...',
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
      // Call REST API
      const response = await fetch(`${API_BASE_URL}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: content,
          session_id: currentSession.id,
          context: []
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`)
      }

      const data = await response.json()

      // Replace loading message with response
      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: data.response || 'I apologize, but I could not generate a response.',
        timestamp: new Date().toISOString(),
        sources: data.sources || [],
        attachments: data.attachments || [],
        processing_time: data.processing_time,
        tokens_generated: data.tokens_generated
      }

      updatedSession = {
        ...updatedSession,
        messages: [...updatedSession.messages.slice(0, -1), assistantMessage],
        updated_at: new Date().toISOString()
      }

      setCurrentSession(updatedSession)
      setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
    } catch (error) {
      console.error('Failed to send message:', error)

      // Show error message
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: `I apologize, but I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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
  }

  const clearSessions = () => {
    setSessions([])
    setCurrentSession(null)
    if (typeof window !== 'undefined') {
      localStorage.removeItem('chat_sessions')
    }
  }

  const value = {
    sessions,
    currentSession,
    isConnected,
    isProcessing,
    createNewSession,
    switchSession,
    sendMessage,
    clearSessions
  }

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