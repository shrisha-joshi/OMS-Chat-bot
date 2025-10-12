'use client'

import React, { useState, useRef, useEffect } from 'react'
import { useChat } from '@/contexts/ChatContext'
import { useAuth } from '@/contexts/AuthContext'
import { Send, Paperclip, Mic, StopCircle, Upload, FileText, Image, Bot, User, ExternalLink, Copy, ThumbsUp, ThumbsDown } from 'lucide-react'

interface Message {
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

export function AdvancedChatInterface() {
  const { currentSession, sendMessage, isProcessing, isConnected } = useChat()
  const { user } = useAuth()
  const [message, setMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [showAttachments, setShowAttachments] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentSession?.messages])

  const handleSendMessage = async () => {
    if (!message.trim() || isProcessing) return
    
    const userMessage = message.trim()
    setMessage('')
    await sendMessage(userMessage)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) {
      // Handle file upload logic
      console.log('Files selected:', files)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <h2 className="text-lg font-semibold text-gray-900">
            {currentSession?.title || 'New Chat Session'}
          </h2>
        </div>
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <span>RAG + Knowledge Graph</span>
          <Bot className="w-4 h-4" />
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {currentSession?.messages.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start a conversation</h3>
            <p className="text-gray-500">Ask me anything about your documents and knowledge base.</p>
          </div>
        ) : (
          currentSession?.messages.map((msg: Message) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-900'} rounded-lg shadow-sm`}>
                <div className="p-4">
                  <div className="flex items-start space-x-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${msg.role === 'user' ? 'bg-blue-500' : 'bg-gray-100'}`}>
                      {msg.role === 'user' ? (
                        <User className="w-4 h-4 text-white" />
                      ) : (
                        <Bot className="w-4 h-4 text-gray-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="font-medium text-sm">
                          {msg.role === 'user' ? user?.email : 'Assistant'}
                        </span>
                        <span className="text-xs opacity-75">
                          {formatTimestamp(msg.timestamp)}
                        </span>
                      </div>
                      <div className="prose prose-sm max-w-none">
                        {msg.content}
                      </div>
                      
                      {/* Sources */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3 p-3 bg-gray-50 rounded-md border">
                          <h4 className="text-xs font-medium text-gray-700 mb-2 flex items-center">
                            <FileText className="w-3 h-3 mr-1" />
                            Sources ({msg.sources.length})
                          </h4>
                          <div className="space-y-2">
                            {msg.sources.map((source, idx) => (
                              <div key={idx} className="text-xs bg-white p-2 rounded border">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-medium text-blue-600">{source.filename}</span>
                                  <span className="text-gray-500">Score: {(source.score * 100).toFixed(1)}%</span>
                                </div>
                                <p className="text-gray-600 line-clamp-2">{source.text}</p>
                                <button className="text-blue-500 hover:text-blue-700 mt-1 flex items-center">
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  View Source
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Attachments */}
                      {msg.attachments && msg.attachments.length > 0 && (
                        <div className="mt-3 space-y-2">
                          {msg.attachments.map((attachment, idx) => (
                            <div key={idx} className="flex items-center space-x-2 p-2 bg-gray-50 rounded border">
                              {attachment.type.startsWith('image') ? (
                                <Image className="w-4 h-4 text-gray-500" />
                              ) : (
                                <FileText className="w-4 h-4 text-gray-500" />
                              )}
                              <span className="text-sm text-gray-700">{attachment.title}</span>
                              <button className="text-blue-500 hover:text-blue-700">
                                <ExternalLink className="w-3 h-3" />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Message Actions */}
                      {msg.role === 'assistant' && (
                        <div className="flex items-center space-x-2 mt-3 pt-2 border-t border-gray-200">
                          <button
                            onClick={() => copyToClipboard(msg.content)}
                            className="text-gray-400 hover:text-gray-600"
                            title="Copy message"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                          <button className="text-gray-400 hover:text-green-600" title="Like">
                            <ThumbsUp className="w-4 h-4" />
                          </button>
                          <button className="text-gray-400 hover:text-red-600" title="Dislike">
                            <ThumbsDown className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg shadow-sm p-4 max-w-xs">
              <div className="flex items-center space-x-3">
                <Bot className="w-6 h-6 text-gray-400" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-gray-500">Processing...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1 min-h-0">
            <div className="relative">
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask anything about your documents..."
                className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '120px' }}
                disabled={isProcessing}
              />
              
              {/* Attachment Button */}
              <button
                onClick={() => setShowAttachments(!showAttachments)}
                className="absolute right-2 top-2 p-2 text-gray-400 hover:text-gray-600"
                title="Attach files"
              >
                <Paperclip className="w-4 h-4" />
              </button>
            </div>

            {/* Attachment Options */}
            {showAttachments && (
              <div className="mt-2 p-3 bg-gray-50 rounded-lg border">
                <div className="flex space-x-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center space-x-2 px-3 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                  >
                    <Upload className="w-4 h-4" />
                    <span className="text-sm">Upload File</span>
                  </button>
                  <button
                    onClick={() => setIsRecording(!isRecording)}
                    className={`flex items-center space-x-2 px-3 py-2 border rounded-md ${
                      isRecording 
                        ? 'bg-red-100 border-red-300 text-red-700' 
                        : 'bg-white border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {isRecording ? (
                      <StopCircle className="w-4 h-4" />
                    ) : (
                      <Mic className="w-4 h-4" />
                    )}
                    <span className="text-sm">
                      {isRecording ? 'Stop Recording' : 'Voice Input'}
                    </span>
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileUpload}
                  multiple
                  accept=".pdf,.docx,.xlsx,.pptx,.txt,.csv,.json,.md"
                  className="hidden"
                />
              </div>
            )}
          </div>

          <button
            onClick={handleSendMessage}
            disabled={!message.trim() || isProcessing}
            className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <span>
            {isConnected ? 'Connected to RAG system' : 'Disconnected'}
          </span>
          <span>Press Enter to send, Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  )
}