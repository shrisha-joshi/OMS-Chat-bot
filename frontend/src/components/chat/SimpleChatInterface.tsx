'use client'

import React from 'react'

export interface ChatInterfaceProps {}

export function ChatInterface() {
  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-900">OMS Chatbot</h1>
          <p className="text-sm text-gray-600">RAG + Knowledge Graph System</p>
        </div>
        
        <div className="flex-1 p-4">
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-blue-600 text-xl">💬</span>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Welcome to OMS Chatbot</h3>
            <p className="text-gray-600 text-sm">
              Your RAG + Knowledge Graph assistant is ready to help with document queries and analysis.
            </p>
          </div>
          
          <div className="space-y-3">
            <button className="w-full py-3 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              Start New Chat
            </button>
            <button className="w-full py-3 px-4 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
              Upload Documents
            </button>
            <button className="w-full py-3 px-4 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
              View Knowledge Graph
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Chat Interface</h2>
            <div className="flex items-center space-x-3">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Connected
              </span>
              <button className="text-gray-400 hover:text-gray-600">
                <span className="text-lg">⚙️</span>
              </button>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col bg-gray-50">
          <div className="flex-1 p-6">
            <div className="text-center py-12">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-blue-600 text-2xl">🤖</span>
              </div>
              <h3 className="text-xl font-medium text-gray-900 mb-2">Start a conversation</h3>
              <p className="text-gray-600 max-w-md mx-auto">
                Ask me anything about your documents, knowledge base, or let me help you with document analysis using our RAG system.
              </p>
              
              <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                <div className="p-4 bg-white rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50">
                  <h4 className="font-medium text-gray-900 mb-2">📄 Document Search</h4>
                  <p className="text-sm text-gray-600">Search through your uploaded documents</p>
                </div>
                <div className="p-4 bg-white rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50">
                  <h4 className="font-medium text-gray-900 mb-2">🧠 Knowledge Graph</h4>
                  <p className="text-sm text-gray-600">Explore entity relationships</p>
                </div>
                <div className="p-4 bg-white rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50">
                  <h4 className="font-medium text-gray-900 mb-2">💡 AI Analysis</h4>
                  <p className="text-sm text-gray-600">Get insights from your data</p>
                </div>
              </div>
            </div>
          </div>

          {/* Input Area */}
          <div className="bg-white border-t border-gray-200 p-4">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-end space-x-3">
                <div className="flex-1">
                  <textarea
                    placeholder="Ask me anything about your documents..."
                    className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={1}
                    style={{ minHeight: '44px', maxHeight: '120px' }}
                  />
                </div>
                <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Send
                </button>
              </div>
              
              <div className="flex items-center justify-between mt-2 text-sm text-gray-500">
                <span>RAG + Knowledge Graph System Ready</span>
                <span>Press Enter to send, Shift+Enter for new line</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}