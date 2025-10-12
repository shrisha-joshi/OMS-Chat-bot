'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { io, Socket } from 'socket.io-client'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Array<{
    id: string
    filename: string
    text: string
    score: number
  }>
  attachments?: Array<{
    type: string
    url: string
    title: string
  }>
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

export function ChatProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  useEffect(() => {
    // Load sessions from localStorage
    const savedSessions = localStorage.getItem('chat_sessions')
    if (savedSessions) {
      const parsedSessions = JSON.parse(savedSessions)
      setSessions(parsedSessions)
      if (parsedSessions.length > 0) {
        setCurrentSession(parsedSessions[0])
      }
    }

    // Initialize WebSocket connection
    initializeWebSocket()

    return () => {
      if (socket) {
        socket.disconnect()
      }
    }
  }, [])

  useEffect(() => {
    // Save sessions to localStorage whenever sessions change
    localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  }, [sessions])

  const initializeWebSocket = () => {
    const wsUrl = (typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_WS_URL : undefined) || 'ws://localhost:8000'
    const newSocket = io(wsUrl, {
      transports: ['websocket'],
      autoConnect: true
    })

    newSocket.on('connect', () => {
      setIsConnected(true)
    })

    newSocket.on('disconnect', () => {
      setIsConnected(false)
    })

    newSocket.on('chat_response', (data) => {
      if (data.type === 'token') {
        // Update current message with new token
        updateCurrentMessage(data.content)
      } else if (data.type === 'sources') {
        // Add sources to current message
        updateCurrentMessageSources(data.sources, data.attachments)
        setIsProcessing(false)
      } else if (data.type === 'error') {
        handleChatError(data.content)
        setIsProcessing(false)
      }
    })

    setSocket(newSocket)
  }

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

  const sendMessage = (content: string) => {
    if (!currentSession || !socket || isProcessing) return

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }

    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
      updated_at: new Date().toISOString(),
      title: currentSession.messages.length === 0 ? content.slice(0, 50) + '...' : currentSession.title
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))

    // Create assistant message placeholder
    const assistantMessage: ChatMessage = {
      id: `msg_${Date.now() + 1}`,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    }

    const sessionWithAssistant = {
      ...updatedSession,
      messages: [...updatedSession.messages, assistantMessage]
    }

    setCurrentSession(sessionWithAssistant)
    setSessions(prev => prev.map(s => s.id === sessionWithAssistant.id ? sessionWithAssistant : s))

    setIsProcessing(true)

    // Send to WebSocket
    socket.emit('chat_message', {
      session_id: currentSession.id,
      message: content
    })
  }

  const updateCurrentMessage = (token: string) => {
    if (!currentSession) return

    const updatedMessages = currentSession.messages.map(msg => {
      if (msg.role === 'assistant' && msg.content === '' || 
          msg.id === currentSession.messages[currentSession.messages.length - 1]?.id) {
        return { ...msg, content: msg.content + token }
      }
      return msg
    })

    const updatedSession = {
      ...currentSession,
      messages: updatedMessages,
      updated_at: new Date().toISOString()
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
  }

  const updateCurrentMessageSources = (sources: any[], attachments: any[]) => {
    if (!currentSession) return

    const updatedMessages = currentSession.messages.map(msg => {
      if (msg.id === currentSession.messages[currentSession.messages.length - 1]?.id) {
        return { ...msg, sources, attachments }
      }
      return msg
    })

    const updatedSession = {
      ...currentSession,
      messages: updatedMessages,
      updated_at: new Date().toISOString()
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
  }

  const handleChatError = (error: string) => {
    if (!currentSession) return

    const errorMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: `I apologize, but I encountered an error: ${error}`,
      timestamp: new Date().toISOString()
    }

    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages.slice(0, -1), errorMessage],
      updated_at: new Date().toISOString()
    }

    setCurrentSession(updatedSession)
    setSessions(prev => prev.map(s => s.id === updatedSession.id ? updatedSession : s))
  }

  const clearSessions = () => {
    setSessions([])
    setCurrentSession(null)
    localStorage.removeItem('chat_sessions')
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