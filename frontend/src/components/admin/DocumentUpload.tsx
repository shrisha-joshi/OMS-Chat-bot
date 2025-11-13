'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Upload, FileText, CheckCircle, AlertCircle, Loader, RefreshCw } from 'lucide-react'
import IngestionStatus from '../IngestionStatus'

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: 'uploading' | 'success' | 'error'
  progress: number
  error?: string
  documentId?: string
}

interface DocumentRecord {
  id: string
  filename: string
  size: number
  uploaded_at: string
  ingest_status: 'pending' | 'processing' | 'complete' | 'failed'
  chunks_count: number
  file_type: string
  error_message?: string
}

export function DocumentUpload() {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [previousDocs, setPreviousDocs] = useState<DocumentRecord[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isLoadingDocs, setIsLoadingDocs] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

  // Load previously uploaded documents on mount
  useEffect(() => {
    loadPreviousDocuments()
  }, [])

  const loadPreviousDocuments = async () => {
    setIsLoadingDocs(true)
    try {
      const response = await fetch(`${API_BASE}/admin/documents/list`)
      if (response.ok) {
        const data = await response.json()
        setPreviousDocs(data.documents || [])
        console.log(`✅ Loaded ${data.documents?.length || 0} previous documents`)
      } else {
        console.warn('Failed to load documents:', response.status)
      }
    } catch (error) {
      console.warn('Could not load documents:', error)
    } finally {
      setIsLoadingDocs(false)
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'complete':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">✓ Complete</span>
      case 'processing':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">⟳ Processing</span>
      case 'pending':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">⏳ Pending</span>
      case 'failed':
        return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">✗ Failed</span>
      default:
        return <span className="text-xs text-gray-600">{status}</span>
    }
  }

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
    } catch {
      return dateStr
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files
    if (selectedFiles) {
      for (const file of Array.from(selectedFiles)) {
        handleFileUpload(file)
      }
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

      // Update file status to success with document ID
      setFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'success', progress: 100, documentId: data.document_id }
            : f
        )
      )

      console.log('File uploaded successfully:', data)
      // Refresh document list after successful upload
      setTimeout(() => loadPreviousDocuments(), 1000)
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

  const handleDragOver = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault()
    e.stopPropagation()

    const droppedFiles = e.dataTransfer.files
    if (droppedFiles) {
      for (const file of Array.from(droppedFiles)) {
        handleFileUpload(file)
      }
    }
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
      <label
        htmlFor="document-upload-input"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer w-full"
      >
        <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-1">
          Drag and drop your documents here
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          or click to browse from your computer
        </p>
        <button
          type="button"
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
          id="document-upload-input"
          type="file"
          multiple
          onChange={handleFileSelect}
          accept=".pdf,.docx,.doc,.txt,.html,.json"
          className="hidden"
          disabled={isUploading}
        />
  </label>

      {/* Upload Status */}
      {files.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-medium text-gray-900">Upload Status</h3>
          {files.map(file => (
            <div key={file.id} className="space-y-3">
              <div
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

              {/* Show ingestion progress for successful uploads */}
              {file.status === 'success' && file.documentId && (
                <IngestionStatus 
                  docId={file.documentId}
                  onComplete={() => {
                    console.log(`Document ${file.documentId} processing completed`);
                  }}
                />
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

      {/* Previously Uploaded Documents */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-gray-900">📂 Previously Uploaded Documents</h3>
          <button
            onClick={loadPreviousDocuments}
            disabled={isLoadingDocs}
            className="inline-flex items-center gap-2 px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoadingDocs ? 'animate-spin' : ''}`} />
            {isLoadingDocs ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        {previousDocs.length === 0 ? (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
            <FileText className="w-12 h-12 mx-auto mb-3 text-gray-400" />
            <p className="text-gray-600">No documents uploaded yet</p>
            <p className="text-sm text-gray-500 mt-1">Upload documents above to get started</p>
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Filename</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Size</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Chunks</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Uploaded</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {previousDocs.map(doc => (
                    <tr key={doc.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-900 font-medium">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-gray-400 flex-shrink-0" />
                          <span className="truncate">{doc.filename}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {formatFileSize(doc.size)}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {getStatusBadge(doc.ingest_status)}
                        {doc.error_message && (
                          <div title={doc.error_message} className="text-xs text-red-600 truncate">
                            {doc.error_message}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {doc.ingest_status === 'complete' ? (
                          <span className="text-green-700 font-medium">{doc.chunks_count}</span>
                        ) : (
                          <span className="text-gray-500">{doc.chunks_count || '-'}</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {formatDate(doc.uploaded_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

