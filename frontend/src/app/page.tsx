'use client'

import React, { useState, useEffect } from 'react'
import { ChatInterfaceReal } from '@/components/chat/ChatInterfaceReal'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'

export default function HomePage() {
  const [mounted, setMounted] = useState(false)
  const [sessionId, setSessionId] = useState<string>('')

  useEffect(() => {
    // Initialize session
    const newSessionId = `session-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`
    setSessionId(newSessionId)
    setMounted(true)
  }, [])

  if (!mounted || !sessionId) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header />
        
        {/* Chat interface */}
        <main className="flex-1 relative overflow-hidden">
          <ChatInterfaceReal sessionId={sessionId} />
        </main>
      </div>
    </div>
  )
}