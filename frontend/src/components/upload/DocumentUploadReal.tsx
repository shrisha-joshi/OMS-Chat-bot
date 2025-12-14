'use client'

import React, { useState, useCallback, useRef } from 'react'
import { Upload, X, CheckCircle, AlertCircle, Loader, File } from 'lucide-react'
import toast from 'react-hot-toast'

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  error?: string
  doc_id?: string
}

interface DocumentUploadProps {
  onUploadComplete?: (files: UploadedFile[]) => void
  maxFiles?: number
  maxSize?: number
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

// Get auth token (you'll need to implement auth)
const getAuthToken = () => {
  if (globalThis.window !== undefined) {
    return localStorage.getItem('auth_token') || 'admin_token'
  }
  return 'admin_token'
}

export function DocumentUploadReal({
  onUploadComplete,
  maxFiles = 10,
  maxSize = 209715200 // 200MB
}: Readonly<DocumentUploadProps>) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isMounted, setIsMounted] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragRef = useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    setIsMounted(true)
  }, [])

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragRef.current) {
      dragRef.current.classList.add('border-blue-500', 'bg-blue-50')
    }
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (dragRef.current) {
      dragRef.current.classList.remove('border-blue-500', 'bg-blue-50')
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const uploadFile = useCallback(async (file: File): Promise<void> => {
    const fileId = `${Date.now()}-${Math.random()}`

    const newFile: UploadedFile = {
      id: fileId,
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0
    }

    setUploadedFiles(prev => [...prev, newFile])

    try {
      // Validate file
      if (file.size > maxSize) {
        throw new Error(`File size exceeds ${maxSize / 1024 / 1024}MB limit`)
      }

      const allowedExtensions = ['pdf', 'docx', 'doc', 'txt', 'json', 'xlsx', 'xls', 'pptx', 'ppt', 'jpg', 'jpeg', 'png', 'gif']
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (!ext || !allowedExtensions.includes(ext)) {
        throw new Error(`File type .${ext} not supported`)
      }

      // Create form data
      const formData = new FormData()
      formData.append('file', file)

      // Upload file
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId ? { ...f, status: 'uploading', progress: 50 } : f
        )
      )

      const response = await fetch(`${API_BASE_URL}/admin/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`
        },
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`)
      }

      const data = await response.json()

      // Update to processing status
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'processing', progress: 75, doc_id: data.doc_id }
            : f
        )
      )

      // Poll for processing status
      let attempts = 0
      const maxAttempts = 30 // 30 seconds max wait
      while (attempts < maxAttempts) {
        const statusResponse = await fetch(
          `${API_BASE_URL}/admin/documents/${data.doc_id}`,
          {
            headers: {
              'Authorization': `Bearer ${getAuthToken()}`
            }
          }
        )

        if (statusResponse.ok) {
          const docData = await statusResponse.json()
          
          if (docData.ingest_status === 'completed') {
            setUploadedFiles(prev =>
              prev.map(f =>
                f.id === fileId ? { ...f, status: 'completed', progress: 100 } : f
              )
            )
            toast.success(`${file.name} processed successfully`)
            break
          } else if (docData.ingest_status === 'failed') {
            throw new Error(docData.error_message || 'Processing failed')
          }
        }

        // Wait before polling again
        await new Promise(resolve => setTimeout(resolve, 1000))
        attempts++
      }

      if (attempts >= maxAttempts) {
        throw new Error('Processing timeout - document may still be processing')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed'
      setUploadedFiles(prev =>
        prev.map(f =>
          f.id === fileId ? { ...f, status: 'error', error: errorMessage } : f
        )
      )
      toast.error(`${file.name}: ${errorMessage}`)
    }
  }, [maxSize])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (dragRef.current) {
      dragRef.current.classList.remove('border-blue-500', 'bg-blue-50')
    }

    const files = Array.from(e.dataTransfer.files)
    if (files.length === 0) return

    if (files.length + uploadedFiles.length > maxFiles) {
      toast.error(`Maximum ${maxFiles} files allowed`)
      return
    }

    setIsUploading(true)
    for (const file of files) {
      await uploadFile(file)
    }
    setIsUploading(false)
    onUploadComplete?.(uploadedFiles)
  }, [uploadedFiles, maxFiles, onUploadComplete, uploadFile])

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    if (files.length + uploadedFiles.length > maxFiles) {
      toast.error(`Maximum ${maxFiles} files allowed`)
      return
    }

    setIsUploading(true)
    for (const file of files) {
      await uploadFile(file)
    }
    setIsUploading(false)
    onUploadComplete?.(uploadedFiles)
  }, [uploadedFiles, maxFiles, onUploadComplete, uploadFile])

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf':
        return '📄'
      case 'docx':
      case 'doc':
        return '📝'
      case 'json':
        return '🔗'
      case 'xlsx':
      case 'xls':
        return '📊'
      case 'pptx':
      case 'ppt':
        return '🎯'
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
        return '🖼️'
      case 'txt':
        return '📃'
      default:
        return '📦'
    }
  }

  if (!isMounted) return null

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div className="space-y-4">
        <div
          ref={dragRef}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer bg-gray-50 hover:bg-blue-50"
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={0}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            disabled={isUploading}
            accept=".pdf,.docx,.doc,.txt,.json,.xlsx,.xls,.pptx,.ppt,.jpg,.jpeg,.png,.gif"
          />

          <div className="flex flex-col items-center justify-center">
            <Upload size={48} className="text-gray-400 mb-3" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Drag and drop your files here
            </h3>
            <p className="text-gray-600 text-sm mb-4">
              or click to browse (PDF, DOCX, JSON, XLSX, PPTX, images)
            </p>
            <button
              type="button"
              disabled={isUploading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center gap-2"
            >
              <Upload size={18} />
              Select Files
            </button>
            <p className="text-xs text-gray-500 mt-4">
              Max file size: {maxSize / 1024 / 1024}MB | Max files: {maxFiles}
            </p>
          </div>
        </div>

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-semibold text-gray-900">
              {uploadedFiles.length === 1 ? '1 file' : `${uploadedFiles.length} files`}
            </h4>

            {uploadedFiles.map(file => (
              <div
                key={file.id}
                className="flex items-center gap-3 p-3 bg-white border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
              >
                {/* File Icon */}
                <span className="text-2xl flex-shrink-0">
                  {getFileIcon(file.name)}
                </span>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 truncate">{file.name}</p>
                  <p className="text-xs text-gray-500">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>

                {/* Progress Bar */}
                {file.status !== 'completed' && file.status !== 'error' && (
                  <div className="w-20 h-1.5 bg-gray-200 rounded-full overflow-hidden flex-shrink-0">
                    <div
                      className="h-full bg-blue-600 transition-all duration-300"
                      style={{ width: `${file.progress}%` }}
                    />
                  </div>
                )}

                {/* Status */}
                <div className="flex items-center gap-2 flex-shrink-0">
                  {file.status === 'uploading' && (
                    <>
                      <Loader className="animate-spin text-blue-600" size={16} />
                      <span className="text-xs text-gray-600">Uploading</span>
                    </>
                  )}
                  {file.status === 'processing' && (
                    <>
                      <Loader className="animate-spin text-yellow-600" size={16} />
                      <span className="text-xs text-gray-600">Processing</span>
                    </>
                  )}
                  {file.status === 'completed' && (
                    <>
                      <CheckCircle className="text-green-600" size={16} />
                      <span className="text-xs text-green-600">Completed</span>
                    </>
                  )}
                  {file.status === 'error' && (
                    <>
                      <AlertCircle className="text-red-600" size={16} />
                      <span className="text-xs text-red-600">Error</span>
                    </>
                  )}
                </div>

                {/* Remove Button */}
                {(file.status === 'completed' || file.status === 'error') && (
                  <button
                    onClick={() => removeFile(file.id)}
                    className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors flex-shrink-0"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Error Messages */}
        {uploadedFiles.some(f => f.status === 'error') && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">
              ⚠️ {uploadedFiles.filter(f => f.status === 'error').length} file(s) failed to process
            </p>
            {uploadedFiles.find(f => f.status === 'error')?.error && (
              <p className="text-xs text-red-600 mt-1">
                {uploadedFiles.find(f => f.status === 'error')?.error}
              </p>
            )}
          </div>
        )}

        {/* Upload Summary */}
        {uploadedFiles.length > 0 && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800">
              {uploadedFiles.filter(f => f.status === 'completed').length} uploaded •{' '}
              {uploadedFiles.filter(f => f.status === 'error').length} errors •{' '}
              {uploadedFiles.filter(f => f.status === 'processing' || f.status === 'uploading').length} processing
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
