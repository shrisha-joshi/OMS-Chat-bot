'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { AdminDashboard } from '@/components/admin/AdminDashboard'
import { DocumentUpload } from '@/components/admin/DocumentUpload'

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'upload'>('dashboard')

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header with back button */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link 
              href="/" 
              className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
            >
              ← Back to Chat
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-t border-gray-200 bg-gray-50 px-6">
          <div className="flex gap-8">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`py-3 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'dashboard'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              📊 Dashboard
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-3 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'upload'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              📤 Upload Documents
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'dashboard' && <AdminDashboard />}
        {activeTab === 'upload' && <DocumentUpload />}
      </main>
    </div>
  )
}
