'use client'

import React, { useState } from 'react'
import { useChat } from '@/contexts/ChatContext'
import { 
  MessageSquare, 
  Plus, 
  Search, 
  MoreVertical, 
  Trash2, 
  Edit3, 
  Archive, 
  Star,
  Clock,
  FileText,
  Users
} from 'lucide-react'

interface ChatSessionManagerProps {
  className?: string
}

export function ChatSessionManager({ className = '' }: ChatSessionManagerProps) {
  const { sessions, currentSession, createNewSession, switchSession, clearSessions } = useChat()
  const [searchTerm, setSearchTerm] = useState('')
  const [showArchived, setShowArchived] = useState(false)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.title.toLowerCase().includes(searchTerm.toLowerCase())
    // In a real app, you'd filter by archived status
    return matchesSearch
  })

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInMs = now.getTime() - date.getTime()
    const diffInHours = diffInMs / (1000 * 60 * 60)
    const diffInDays = diffInMs / (1000 * 60 * 60 * 24)

    if (diffInHours < 1) {
      return 'Just now'
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`
    } else if (diffInDays < 7) {
      return `${Math.floor(diffInDays)}d ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const getSessionStats = (session: any) => {
    return {
      messageCount: session.messages?.length || 0,
      lastActivity: session.updated_at || session.created_at,
      hasAttachments: session.messages?.some((msg: any) => msg.attachments?.length > 0) || false
    }
  }

  const handleEditSession = (sessionId: string, currentTitle: string) => {
    setEditingSessionId(sessionId)
    setEditTitle(currentTitle)
  }

  const handleSaveEdit = () => {
    // In a real app, you'd call an API to update the session title
    console.log('Saving session title:', editTitle)
    setEditingSessionId(null)
    setEditTitle('')
  }

  const handleDeleteSession = (sessionId: string) => {
    if (window.confirm('Are you sure you want to delete this session?')) {
      // In a real app, you'd call an API to delete the session
      console.log('Deleting session:', sessionId)
    }
  }

  return (
    <div className={`bg-white border-r border-gray-200 flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-gray-900">Chat Sessions</h2>
          <button
            onClick={createNewSession}
            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            title="New Chat"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search sessions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Quick Stats */}
        <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
          <span>{filteredSessions.length} sessions</span>
          <button
            onClick={() => setShowArchived(!showArchived)}
            className="hover:text-gray-700"
          >
            {showArchived ? 'Hide archived' : 'Show archived'}
          </button>
        </div>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto">
        {filteredSessions.length === 0 ? (
          <div className="p-8 text-center">
            <MessageSquare className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">
              {searchTerm ? 'No sessions match your search' : 'No chat sessions yet'}
            </p>
            {!searchTerm && (
              <button
                onClick={createNewSession}
                className="mt-2 text-blue-600 hover:text-blue-700 text-sm"
              >
                Start your first chat
              </button>
            )}
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {filteredSessions.map((session) => {
              const stats = getSessionStats(session)
              const isActive = currentSession?.id === session.id
              const isEditing = editingSessionId === session.id

              return (
                <div
                  key={session.id}
                  className={`group relative p-3 rounded-lg cursor-pointer transition-colors ${
                    isActive 
                      ? 'bg-blue-50 border border-blue-200' 
                      : 'hover:bg-gray-50 border border-transparent'
                  }`}
                  onClick={() => !isEditing && switchSession(session.id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      {isEditing ? (
                        <div className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="flex-1 text-sm font-medium border border-gray-300 rounded px-2 py-1"
                            autoFocus
                            onKeyPress={(e) => e.key === 'Enter' && handleSaveEdit()}
                            onBlur={handleSaveEdit}
                          />
                        </div>
                      ) : (
                        <h3 className={`text-sm font-medium truncate ${
                          isActive ? 'text-blue-900' : 'text-gray-900'
                        }`}>
                          {session.title}
                        </h3>
                      )}

                      <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
                        <div className="flex items-center space-x-1">
                          <MessageSquare className="w-3 h-3" />
                          <span>{stats.messageCount}</span>
                        </div>
                        
                        <div className="flex items-center space-x-1">
                          <Clock className="w-3 h-3" />
                          <span>{formatDate(stats.lastActivity)}</span>
                        </div>

                        {stats.hasAttachments && (
                          <div className="flex items-center space-x-1">
                            <FileText className="w-3 h-3" />
                          </div>
                        )}
                      </div>

                      {/* Preview of last message */}
                      {session.messages && session.messages.length > 0 && (
                        <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                          {session.messages[session.messages.length - 1].content.slice(0, 100)}...
                        </p>
                      )}
                    </div>

                    {/* Actions Menu */}
                    <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          // Toggle star
                        }}
                        className="p-1 text-gray-400 hover:text-yellow-500 rounded"
                        title="Star session"
                      >
                        <Star className="w-3 h-3" />
                      </button>

                      <div className="relative">
                        <button
                          onClick={(e) => e.stopPropagation()}
                          className="p-1 text-gray-400 hover:text-gray-600 rounded"
                          title="More options"
                        >
                          <MoreVertical className="w-3 h-3" />
                        </button>

                        {/* Dropdown Menu (would need proper implementation) */}
                        <div className="hidden absolute right-0 mt-1 w-32 bg-white border border-gray-200 rounded-md shadow-lg z-10">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleEditSession(session.id, session.title)
                            }}
                            className="w-full px-3 py-2 text-left text-xs text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
                          >
                            <Edit3 className="w-3 h-3" />
                            <span>Rename</span>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              // Archive session
                            }}
                            className="w-full px-3 py-2 text-left text-xs text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
                          >
                            <Archive className="w-3 h-3" />
                            <span>Archive</span>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteSession(session.id)
                            }}
                            className="w-full px-3 py-2 text-left text-xs text-red-600 hover:bg-red-50 flex items-center space-x-2"
                          >
                            <Trash2 className="w-3 h-3" />
                            <span>Delete</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Session Tags */}
                  <div className="flex items-center space-x-1 mt-2">
                    {isActive && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-blue-100 text-blue-800">
                        Active
                      </span>
                    )}
                    {stats.hasAttachments && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600">
                        Has files
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
          <span>Total: {sessions.length} sessions</span>
          <button
            onClick={clearSessions}
            className="text-red-600 hover:text-red-700"
          >
            Clear all
          </button>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center space-x-2">
          <button className="flex-1 py-2 px-3 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors">
            Export All
          </button>
          <button className="flex-1 py-2 px-3 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors">
            Import
          </button>
        </div>
      </div>
    </div>
  )
}