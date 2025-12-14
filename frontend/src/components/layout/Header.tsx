'use client'

import { useRouter } from 'next/navigation'
import SystemStatus from './SystemStatus'

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HeaderProps {}

export function Header() {
  const router = useRouter()

  const handleAdminClick = () => {
    router.push('/admin')
  }

  return (
    <header style={{ padding: '16px 24px', background: 'white', borderBottom: '1px solid #e9ecef', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div className="flex items-center gap-4">
        <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 600 }}>OMS Chatbot</h1>
        <SystemStatus />
      </div>
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <button 
          onClick={handleAdminClick}
          style={{ 
            padding: '8px 16px', 
            background: '#007bff', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px', 
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 500,
            transition: 'background 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = '#0056b3'}
          onMouseLeave={(e) => e.currentTarget.style.background = '#007bff'}
        >
          📁 Admin Dashboard
        </button>
      </div>
    </header>
  )
}