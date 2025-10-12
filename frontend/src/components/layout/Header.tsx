export interface HeaderProps {}

export function Header() {
  return (
    <header style={{ padding: '16px 24px', background: 'white', borderBottom: '1px solid #e9ecef', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 600 }}>OMS Chatbot</h1>
      <div>
        <button style={{ padding: '8px 16px', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
          Admin
        </button>
      </div>
    </header>
  )
}