import { ReactElement, ReactNode } from 'react'

declare global {
  namespace JSX {
    interface IntrinsicElements {
      [elemName: string]: any
    }
  }
}

declare module '*.css' {
  const content: { [className: string]: string }
  export default content
}

declare module '*.scss' {
  const content: { [className: string]: string }
  export default content
}

// Chat and Media Types
declare namespace Chat {
  interface Source {
    id?: string
    doc_id?: string
    chunk_id?: string
    filename: string
    similarity: number
    text: string
    page?: number
    source_label?: string
    score?: number
  }

  interface MediaAttachment {
    type: 'image' | 'video' | 'pdf' | 'youtube' | 'link'
    url?: string
    videoId?: string
    filename?: string
    title?: string
    data?: string  // Base64 encoded for images
    page?: number
    relevance?: number
  }

  interface ValidationDetails {
    is_valid: boolean
    validation_score: number
    has_citations: boolean
    citation_count: number
    has_generic_phrases: boolean
    generic_phrase_count?: number
    warnings?: string[]
    issues?: string[]
  }

  interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    sources?: Source[]
    attachments?: MediaAttachment[]
    media_suggestions?: MediaAttachment[]
    validation_details?: ValidationDetails
    processing_time?: number
    tokens_generated?: number
  }

  interface ChatResponse {
    response: string
    sources: Source[]
    attachments: MediaAttachment[]
    media_suggestions?: MediaAttachment[]
    validation_details?: ValidationDetails
    session_id: string
    processing_time: number
    tokens_generated: number
    phase3_metrics?: Record<string, any>
  }

  interface ChatRequest {
    query: string
    session_id?: string
    context?: Message[]
    stream?: boolean
  }
}

// Extend the global process interface for browser environments
declare global {
  interface Window {
    process?: {
      env: {
        [key: string]: string | undefined
        NEXT_PUBLIC_API_URL?: string
        NEXT_PUBLIC_API_BASE?: string
        NODE_ENV?: string
      }
    }
  }
  
  var process: {
    env: {
      [key: string]: string | undefined
      NEXT_PUBLIC_API_URL?: string
      NEXT_PUBLIC_API_BASE?: string
      NODE_ENV?: string
    }
  }
}

export {}