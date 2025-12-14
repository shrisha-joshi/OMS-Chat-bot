'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Paperclip, Loader } from 'lucide-react'
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
  validation_details?: {
    is_valid: boolean
    validation_score: number
    has_citations: boolean
    citation_count: number
    has_generic_phrases: boolean
  }
  processing_time?: number
  tokens_generated?: number
}

// interface ChatSession {
//   id: string
//   title: string
//   messages: Message[]
//   created_at: string
// }

interface ChatInterfaceProps {
  sessionId: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

export function ChatInterfaceReal({ sessionId: propSessionId }: Readonly<ChatInterfaceProps>) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>(propSessionId)
  const [isMounted, setIsMounted] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Initialize session and load history
  useEffect(() => {
    setIsMounted(true)
    setSessionId(propSessionId)
    
    // Load previous messages from API
    const loadSessionHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/chat/history/${propSessionId}`)
        if (response.ok) {
          const data = await response.json()
          if (data.messages && Array.isArray(data.messages) && data.messages.length > 0) {
            setMessages(data.messages)
            console.info(`✅ Loaded ${data.messages.length} messages from session history`)
          }
        }
      } catch (error) {
        console.warn('Could not load history:', error)
        // Graceful degradation - continue with empty chat
      }
    }
    
    loadSessionHistory()
  }, [propSessionId])

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
        const errorData = await response.json().catch(() => ({}))
        const errorMsg = errorData.detail || errorData.error || response.statusText
        throw new Error(`API error (${response.status}): ${errorMsg}`)
      }

      const data = await response.json()

      // Check if response is empty or error
      if (!data.response) {
        throw new Error('Empty response from LLM. The service may be experiencing issues.')
      }

      const assistantMessage: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        sources: data.sources,
        attachments: data.media_suggestions || data.attachments,  // Include media suggestions
        validation_details: data.validation_details,  // Include validation score
        processing_time: data.processing_time,
        tokens_generated: data.tokens_generated
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Show validation score if available
      if (data.validation_details?.is_valid) {
        const score = (data.validation_details.validation_score * 100).toFixed(0)
        toast.success(`✅ Response valid (Score: ${score}%) | ${data.tokens_generated || 0} tokens`)
      } else {
        toast.success(`✅ Response received (${data.tokens_generated || 0} tokens)`)
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      console.error('Chat error:', error)
      
      // Show detailed error message
      const errorDetails = errorMessage.includes('API error') 
        ? errorMessage 
        : `Failed to send message: ${errorMessage}`
      
      toast.error(errorDetails, {
        duration: 5000
      })
      
      // Add error message to chat for visibility
      const errorMessage_obj: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: `❌ Error: ${errorMessage}\n\nPlease try again or check the backend logs for details.`,
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMessage_obj])
    } finally {
      setIsLoading(false)
    }
  }, [input, sessionId])

  // Handle key press
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Render media inline within message (like ChatGPT, Discord, WhatsApp)
  const renderInlineMedia = (attachment: Message['attachments'][0], index: number) => {
    switch (attachment.type) {
      case 'youtube':
        return (
          <div key={`yt-${index}`} className="my-3 rounded-lg overflow-hidden bg-black max-w-xl hover:shadow-lg transition-shadow group">
            <div className="relative pt-[56.25%]">
              <iframe
                src={`https://www.youtube.com/embed/${attachment.videoId}?modestbranding=1&rel=0`}
                title={attachment.title || 'YouTube Video'}
                style={{ border: 0 }}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="absolute inset-0 w-full h-full"
              />
            </div>
            {attachment.title && (
              <div className="p-2 bg-gray-900 text-white text-xs">
                <p className="truncate">🎥 {attachment.title}</p>
              </div>
            )}
          </div>
        )
      case 'image':
        return (
          <div key={`img-${index}`} className="my-3 rounded-lg overflow-hidden max-w-xl hover:shadow-lg transition-shadow group">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={attachment.url}
              alt={attachment.filename || 'Image'}
              className="w-full h-auto rounded-lg border border-gray-200 dark:border-gray-700 cursor-pointer hover:opacity-90 transition-opacity"
              onError={(e) => {
                e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"%3E%3Cpath stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /%3E%3C/svg%3E'
              }}
            />
            {attachment.filename && (
              <div className="p-2 bg-gray-100 dark:bg-gray-800 text-xs text-gray-700 dark:text-gray-300">
                📷 {attachment.filename}
              </div>
            )}
          </div>
        )
      case 'pdf':
        return (
          <a
            key={`pdf-${index}`}
            href={attachment.url}
            target="_blank"
            rel="noopener noreferrer"
            className="my-2 inline-flex items-center px-4 py-2 bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 text-red-700 dark:text-red-300 rounded-lg hover:from-red-100 hover:to-red-200 transition-all border border-red-200 dark:border-red-700 group"
          >
            <span className="text-lg mr-2 group-hover:scale-110 transition-transform">📄</span>
            <span className="font-medium text-sm">{attachment.filename || 'PDF Document'}</span>
            <span className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity text-xs">↗</span>
          </a>
        )
      case 'link':
        return (
          <a
            key={`link-${index}`}
            href={attachment.url}
            target="_blank"
            rel="noopener noreferrer"
            className="my-2 inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 text-blue-700 dark:text-blue-300 rounded-lg hover:from-blue-100 hover:to-blue-200 transition-all border border-blue-200 dark:border-blue-700 group"
          >
            <span className="text-lg mr-2 group-hover:scale-110 transition-transform">🔗</span>
            <span className="font-medium text-sm truncate">{attachment.title || 'External Link'}</span>
            <span className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity text-xs">↗</span>
          </a>
        )
      default:
        return null
    }
  }

  // Render message content with inline media parsing
  const renderMessageContent = (message: Message) => {
    const content = message.content || ''
    
    // Don't parse for error messages
    if (content.includes('❌ Error:')) {
      return <p className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap">{content}</p>
    }

    return (
      <div className="space-y-2">
        <p className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap leading-relaxed">{content}</p>
      </div>
    )
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
                  {/* Message content (with inline media) */}
                  {message.role === 'user' ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  ) : (
                    renderMessageContent(message)
                  )}

                  {/* Render media inline right after content */}
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {message.attachments.map((attachment, idx) =>
                        renderInlineMedia(attachment, idx)
                      )}
                    </div>
                  )}

                  {/* Render sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className={`mt-3 pt-3 border-t ${message.role === 'user' ? 'border-blue-500' : 'border-gray-200'}`}>
                      <p className={`text-xs font-semibold mb-2 ${message.role === 'user' ? 'text-blue-100' : 'text-gray-600'}`}>
                        📚 Sources:
                      </p>
                      <div className="space-y-1">
                        {message.sources.map((source, idx) => (
                          <div
                            key={`${message.id}-source-${idx}`}
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

                  {/* Validation Score */}
                  {message.validation_details && (
                    <div className={`mt-3 pt-3 border-t ${message.role === 'user' ? 'border-blue-500' : 'border-green-200'}`}>
                      <div className={`flex items-center justify-between ${message.role === 'user' ? 'text-blue-100' : 'text-green-700'}`}>
                        <span className="text-xs font-semibold">✅ Response Quality</span>
                        <span className={`text-sm font-bold ${message.validation_details.validation_score >= 0.7 ? 'text-green-600' : 'text-yellow-600'}`}>
                          {(message.validation_details.validation_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="mt-1 text-xs space-y-0.5">
                        <p>📌 Citations: {message.validation_details.citation_count}</p>
                        <p>{message.validation_details.has_citations ? '✓' : '✗'} Document-based</p>
                      </div>
                    </div>
                  )}

                  {/* Metadata */}
                  {Boolean(message.processing_time) && (
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
                onKeyDown={handleKeyDown}
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
