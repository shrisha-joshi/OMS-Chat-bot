'use client'

import React, { useState, useRef } from 'react'
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from 'lucide-react'

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: 'uploading' | 'success' | 'error'
  progress: number
  error?: string
}

export function DocumentUpload() {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files
    if (selectedFiles) {
      Array.from(selectedFiles).forEach(file => {
        handleFileUpload(file)
      })
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleFileUpload = async (file: File) => {
    const fileId = `${Date.now()}-${Math.random()}`
    const newFile: UploadedFile = {
      id: fileId,
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0
    }

    setFiles(prev => [...prev, newFile])
    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      // No authentication required - public endpoint
      const response = await fetch(`${API_BASE}/admin/documents/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`)
      }

      const data = await response.json()

      // Update file status to success
      setFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'success', progress: 100 }
            : f
        )
      )

      console.log('File uploaded successfully:', data)
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      
      setFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { 
                ...f, 
                status: 'error', 
                error: errorMessage,
                progress: 0
              }
            : f
        )
      )

      console.error('Upload error:', error)
    } finally {
      setIsUploading(false)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()

    const droppedFiles = e.dataTransfer.files
    if (droppedFiles) {
      Array.from(droppedFiles).forEach(file => {
        handleFileUpload(file)
      })
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      {/* Upload Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-medium text-blue-900 mb-2">📝 How to Upload Documents</h3>
        <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
          <li>Supported formats: PDF, DOCX, TXT, HTML, JSON</li>
          <li>Maximum file size: 50 MB</li>
          <li>Drag and drop files or click to browse</li>
          <li>After upload, documents will be processed and indexed automatically</li>
        </ul>
      </div>

      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-1">
          Drag and drop your documents here
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          or click to browse from your computer
        </p>
        <button
          className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium text-sm"
          onClick={(e) => {
            e.stopPropagation()
            fileInputRef.current?.click()
          }}
        >
          Browse Files
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          accept=".pdf,.docx,.doc,.txt,.html,.json"
          className="hidden"
          disabled={isUploading}
        />
      </div>

      {/* Upload Status */}
      {files.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-medium text-gray-900">Upload Status</h3>
          {files.map(file => (
            <div
              key={file.id}
              className="bg-white border border-gray-200 rounded-lg p-4 flex items-center gap-4"
            >
              <FileText className="w-8 h-8 text-gray-400 flex-shrink-0" />
              
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(file.size)}
                </p>
              </div>

              {file.status === 'uploading' && (
                <div className="flex items-center gap-2">
                  <Loader className="w-4 h-4 text-blue-600 animate-spin" />
                  <span className="text-sm text-gray-600">{file.progress}%</span>
                </div>
              )}

              {file.status === 'success' && (
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
              )}

              {file.status === 'error' && (
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                  <div className="text-xs text-red-600 max-w-xs">
                    {file.error}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Processing Info */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="font-medium text-green-900 mb-2">✅ What Happens Next?</h3>
        <ol className="text-sm text-green-800 space-y-1 list-decimal list-inside">
          <li>Documents are uploaded to MongoDB storage</li>
          <li>Text is extracted and split into chunks (750 tokens each)</li>
          <li>Each chunk is converted to vector embeddings (384-dimensional)</li>
          <li>Vectors are stored in Qdrant for fast similarity search</li>
          <li>Knowledge graph relationships are created in ArangoDB</li>
          <li>Documents become searchable in chat within seconds</li>
        </ol>
      </div>
    </div>
  )
}
