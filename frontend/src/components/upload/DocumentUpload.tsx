'use client'

import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Image as ImageIcon, File, X, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import toast from 'react-hot-toast'

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  error?: string
}

interface DocumentUploadProps {
  onUploadComplete?: (files: UploadedFile[]) => void
  maxFiles?: number
  maxSize?: number
}

export function DocumentUpload({ onUploadComplete, maxFiles = 10, maxSize = 209715200 }: DocumentUploadProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    setIsUploading(true)
    const newFiles: UploadedFile[] = acceptedFiles.map((file, index) => ({
      id: `${Date.now()}-${index}`,
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])

    // Simulate file upload and processing
    for (const [index, file] of acceptedFiles.entries()) {
      const fileId = newFiles[index].id
      
      try {
        // Upload simulation
        await simulateUpload(fileId)
        
        // Processing simulation
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileId 
              ? { ...f, status: 'processing', progress: 0 }
              : f
          )
        )
        
        await simulateProcessing(fileId)
        
        // Complete
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileId 
              ? { ...f, status: 'completed', progress: 100 }
              : f
          )
        )
        
        toast.success(`${file.name} processed successfully`)
      } catch {
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileId 
              ? { ...f, status: 'error', error: 'Processing failed' }
              : f
          )
        )
        toast.error(`Failed to process ${file.name}`)
      }
    }

    setIsUploading(false)
    onUploadComplete?.(uploadedFiles)
  }, [uploadedFiles, onUploadComplete])

  const simulateUpload = (fileId: string): Promise<void> => {
    return new Promise((resolve) => {
      let progress = 0
      const interval = setInterval(() => {
        progress += Math.random() * 30
        if (progress >= 100) {
          progress = 100
          clearInterval(interval)
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, progress: 100 }
                : f
            )
          )
          resolve()
        } else {
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, progress: Math.round(progress) }
                : f
            )
          )
        }
      }, 200)
    })
  }

  const simulateProcessing = (fileId: string): Promise<void> => {
    return new Promise((resolve) => {
      let progress = 0
      const interval = setInterval(() => {
        progress += Math.random() * 25
        if (progress >= 100) {
          progress = 100
          clearInterval(interval)
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, progress: 100 }
                : f
            )
          )
          resolve()
        } else {
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, progress: Math.round(progress) }
                : f
            )
          )
        }
      }, 300)
    })
  }

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <ImageIcon className="w-5 h-5" />
    if (type.includes('pdf')) return <FileText className="w-5 h-5 text-red-500" />
    if (type.includes('word')) return <FileText className="w-5 h-5 text-blue-500" />
    if (type.includes('excel') || type.includes('spreadsheet')) return <FileText className="w-5 h-5 text-green-500" />
    if (type.includes('powerpoint') || type.includes('presentation')) return <FileText className="w-5 h-5 text-orange-500" />
    return <File className="w-5 h-5 text-gray-500" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-powerpoint': ['.ppt'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'text/markdown': ['.md'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    },
    maxFiles,
    maxSize,
    disabled: isUploading
  })

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Documents</h3>
          <p className="text-gray-600">
            Upload documents to add them to your knowledge base. Supported formats: PDF, Word, Excel, PowerPoint, Text, Images
          </p>
        </div>

        {/* Drop Zone */}
        <div className="p-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-400 bg-blue-50'
                : isUploading
                ? 'border-gray-300 bg-gray-50 cursor-not-allowed'
                : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragActive ? 'text-blue-500' : 'text-gray-400'}`} />
            
            {isDragActive ? (
              <p className="text-blue-600 font-medium">Drop the files here...</p>
            ) : isUploading ? (
              <p className="text-gray-500">Processing files...</p>
            ) : (
              <div>
                <p className="text-gray-900 font-medium mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-gray-500 text-sm">
                  Max {maxFiles} files, up to {formatFileSize(maxSize)} each
                </p>
              </div>
            )}
          </div>

          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                Files ({uploadedFiles.length})
              </h4>
              <div className="space-y-3">
                {uploadedFiles.map((file) => (
                  <div key={file.id} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg border">
                    <div className="flex-shrink-0">
                      {getFileIcon(file.type)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {file.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatFileSize(file.size)}
                      </p>
                      
                      {/* Progress Bar */}
                      {(file.status === 'uploading' || file.status === 'processing') && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-gray-600">
                              {file.status === 'uploading' ? 'Uploading...' : 'Processing...'}
                            </span>
                            <span className="text-gray-600">{file.progress}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${file.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className="flex-shrink-0 flex items-center space-x-2">
                      {file.status === 'uploading' || file.status === 'processing' ? (
                        <Loader className="w-4 h-4 text-blue-500 animate-spin" />
                      ) : file.status === 'completed' ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : file.status === 'error' ? (
                        <div title={file.error}>
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        </div>
                      ) : null}
                      
                      <button
                        onClick={() => removeFile(file.id)}
                        className="text-gray-400 hover:text-red-500"
                        title="Remove file"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Upload Stats */}
          {uploadedFiles.length > 0 && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between text-sm">
                <span className="text-blue-800">
                  {uploadedFiles.filter(f => f.status === 'completed').length} of {uploadedFiles.length} files processed
                </span>
                <span className="text-blue-600">
                  Ready for knowledge base integration
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}