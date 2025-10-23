'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Paperclip, Loader, AlertCircle, CheckCircle, Copy, Download } from 'lucide-react'
import toast from 'react-hot-toast'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Array<{
    filename: string
    similarity: number
    text: string
    page?: number
  }>
  attachments?: Array<{
    type: 'image' | 'video' | 'pdf' | 'youtube' | 'link'
    url?: string
    videoId?: string
    filename?: string
    title?: string
  }>
  processing_time?: number
  tokens_generated?: number
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
  created_at: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

export function ChatInterfaceReal() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')
  const [isMounted, setIsMounted] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Initialize session
  useEffect(() => {
    setIsMounted(true)
    const newSessionId = `session-${Date.now()}`
    setSessionId(newSessionId)
  }, [])

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Handle sending message
  const handleSendMessage = useCallback(async () => {
    if (!input.trim() || !sessionId) return

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          session_id: sessionId,
          stream: false
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`)
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        sources: data.sources,
        attachments: data.attachments,
        processing_time: data.processing_time,
        tokens_generated: data.tokens_generated
      }

      setMessages(prev => [...prev, assistantMessage])
      toast.success('Response received!')
    } catch (error) {
      toast.error(`Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`)
      console.error('Chat error:', error)
    } finally {
      setIsLoading(false)
    }
  }, [input, sessionId])

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Render media based on type
  const renderAttachment = (attachment: Message['attachments'][0]) => {
    switch (attachment.type) {
      case 'youtube':
        return (
          <div key={attachment.videoId} className="mt-2 rounded-lg overflow-hidden bg-black max-w-sm">
            <iframe
              width="100%"
              height="250"
              src={`https://www.youtube.com/embed/${attachment.videoId}`}
              title={attachment.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="w-full"
            />
          </div>
        )
      case 'image':
        return (
          <div key={attachment.url} className="mt-2 rounded-lg overflow-hidden max-w-sm">
            <img
              src={attachment.url}
              alt={attachment.filename}
              className="w-full h-auto rounded-lg border border-gray-200"
            />
          </div>
        )
      case 'pdf':
        return (
          <a
            key={attachment.url}
            href={attachment.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
          >
            <span className="text-lg mr-2">📄</span>
            {attachment.filename}
          </a>
        )
      case 'link':
        return (
          <a
            key={attachment.url}
            href={attachment.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
          >
            <span className="text-lg mr-2">🔗</span>
            {attachment.title || 'External Link'}
          </a>
        )
      default:
        return null
    }
  }

  if (!isMounted) return null

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">OMS Chatbot</h1>
          <p className="text-sm text-gray-600">RAG + Knowledge Graph</p>
          <div className="mt-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded inline-block">
            ✓ Connected
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <button className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
            <span>+</span> New Chat
          </button>

          {messages.length > 0 && (
            <div className="mt-6">
              <h3 className="text-xs font-semibold text-gray-600 uppercase mb-3">Session</h3>
              <p className="text-xs text-gray-500 mb-2">{messages.length} messages</p>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-gray-200 space-y-2">
          <button className="w-full py-2 px-4 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            📚 Documentation
          </button>
          <button className="w-full py-2 px-4 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            ⚙️ Settings
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Chat Assistant</h2>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">Session: {sessionId.slice(0, 8)}...</span>
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-3xl">🤖</span>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Start a Conversation</h3>
                <p className="text-gray-600 max-w-md">
                  Ask me anything about your documents. I can search, analyze, and provide insights using RAG and Knowledge Graph technology.
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white rounded-lg rounded-tr-none'
                      : 'bg-white text-gray-900 rounded-lg rounded-tl-none border border-gray-200'
                  } p-4`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>

                  {/* Render attachments */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {message.attachments.map((attachment, idx) => (
                        <div key={idx}>
                          {renderAttachment(attachment)}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Render sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className={`mt-3 pt-3 border-t ${message.role === 'user' ? 'border-blue-500' : 'border-gray-200'}`}>
                      <p className={`text-xs font-semibold mb-2 ${message.role === 'user' ? 'text-blue-100' : 'text-gray-600'}`}>
                        Sources:
                      </p>
                      <div className="space-y-1">
                        {message.sources.map((source, idx) => (
                          <div
                            key={idx}
                            className={`text-xs p-2 rounded ${
                              message.role === 'user'
                                ? 'bg-blue-500 bg-opacity-50'
                                : 'bg-gray-100'
                            }`}
                          >
                            <p className={`font-medium ${message.role === 'user' ? 'text-blue-100' : 'text-gray-700'}`}>
                              {source.filename}
                            </p>
                            <p className={`${message.role === 'user' ? 'text-blue-50' : 'text-gray-600'}`}>
                              Similarity: {(source.similarity * 100).toFixed(1)}%
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Metadata */}
                  {message.processing_time && (
                    <div className={`text-xs mt-2 ${message.role === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                      ⏱️ {message.processing_time.toFixed(2)}s | 🔢 {message.tokens_generated} tokens
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-end gap-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="Attach file"
              >
                <Paperclip size={20} />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".pdf,.docx,.txt,.json,.xlsx,.pptx,.jpg,.png,.gif"
              />

              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about your documents..."
                className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={1}
                disabled={isLoading}
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />

              <button
                onClick={handleSendMessage}
                disabled={isLoading || !input.trim()}
                className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                {isLoading ? <Loader className="animate-spin" size={20} /> : <Send size={20} />}
              </button>
            </div>

            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span>Press Enter to send, Shift+Enter for new line</span>
              {isLoading && <span className="flex items-center gap-1">
                <Loader className="animate-spin" size={14} />
                Processing...
              </span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
