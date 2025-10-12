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

// Extend the global process interface for browser environments
declare global {
  interface Window {
    process?: {
      env: {
        [key: string]: string | undefined
        NEXT_PUBLIC_API_URL?: string
        NODE_ENV?: string
      }
    }
  }
  
  var process: {
    env: {
      [key: string]: string | undefined
      NEXT_PUBLIC_API_URL?: string
      NODE_ENV?: string
    }
  }
}

export {}