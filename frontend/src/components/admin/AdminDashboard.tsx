'use client'

import React, { useState, useEffect } from 'react'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  Users, 
  MessageSquare, 
  FileText, 
  Database, 
  Activity, 
  TrendingUp,
  Server,
  Clock,
  Download,
  RefreshCw
} from 'lucide-react'

interface DashboardStats {
  totalUsers: number
  totalChats: number
  totalDocuments: number
  vectorsStored: number
  avgResponseTime: number
  systemUptime: number
}

interface ChartData {
  name: string
  value: number
  color?: string
}

interface AdminDashboardProps {
  className?: string
}

export function AdminDashboard({ className = '' }: AdminDashboardProps) {
  const [stats, setStats] = useState<DashboardStats>({
    totalUsers: 0,
    totalChats: 0,
    totalDocuments: 0,
    vectorsStored: 0,
    avgResponseTime: 0,
    systemUptime: 0
  })

  const [isLoading, setIsLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d')

  // Sample data for demonstration
  const chatVolumeData = [
    { name: 'Mon', chats: 45, responses: 42 },
    { name: 'Tue', chats: 67, responses: 61 },
    { name: 'Wed', chats: 89, responses: 85 },
    { name: 'Thu', chats: 123, responses: 118 },
    { name: 'Fri', chats: 145, responses: 139 },
    { name: 'Sat', chats: 98, responses: 92 },
    { name: 'Sun', chats: 76, responses: 71 }
  ]

  const responseTimeData = [
    { name: '00:00', time: 1.2 },
    { name: '04:00', time: 0.9 },
    { name: '08:00', time: 2.3 },
    { name: '12:00', time: 3.1 },
    { name: '16:00', time: 2.8 },
    { name: '20:00', time: 1.7 }
  ]

  const documentTypesData: ChartData[] = [
    { name: 'PDF', value: 145, color: '#3B82F6' },
    { name: 'Word', value: 89, color: '#10B981' },
    { name: 'Excel', value: 34, color: '#F59E0B' },
    { name: 'PowerPoint', value: 23, color: '#EF4444' },
    { name: 'Text', value: 67, color: '#8B5CF6' }
  ]

  const knowledgeBaseData = [
    { name: 'Documents', value: 358 },
    { name: 'Entities', value: 1247 },
    { name: 'Relations', value: 3891 },
    { name: 'Concepts', value: 892 }
  ]

  useEffect(() => {
    // Simulate loading stats
    const loadStats = async () => {
      setIsLoading(true)
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      setStats({
        totalUsers: 247,
        totalChats: 1893,
        totalDocuments: 358,
        vectorsStored: 45678,
        avgResponseTime: 2.3,
        systemUptime: 99.7
      })
      
      setIsLoading(false)
    }

    loadStats()
  }, [selectedTimeRange])

  const formatUptime = (uptime: number) => {
    return `${uptime}%`
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const StatCard = ({ 
    title, 
    value, 
    icon: Icon, 
    change, 
    color = 'blue' 
  }: {
    title: string
    value: string | number
    icon: any
    change?: string
    color?: string
  }) => (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {isLoading ? '...' : value}
          </p>
          {change && (
            <p className="text-sm text-green-600 flex items-center mt-1">
              <TrendingUp className="w-3 h-3 mr-1" />
              {change}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg bg-${color}-100`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  )

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="text-gray-600">Overview of your RAG chatbot system</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          
          <button
            onClick={() => window.location.reload()}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Users"
          value={formatNumber(stats.totalUsers)}
          icon={Users}
          change="+12% from last week"
          color="blue"
        />
        <StatCard
          title="Total Chats"
          value={formatNumber(stats.totalChats)}
          icon={MessageSquare}
          change="+8% from last week"
          color="green"
        />
        <StatCard
          title="Documents"
          value={formatNumber(stats.totalDocuments)}
          icon={FileText}
          change="+23 new documents"
          color="purple"
        />
        <StatCard
          title="Vectors Stored"
          value={formatNumber(stats.vectorsStored)}
          icon={Database}
          change="+2.4K new vectors"
          color="orange"
        />
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">System Health</h3>
            <Activity className="w-5 h-5 text-gray-500" />
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Response Time</span>
                <span>{stats.avgResponseTime}s avg</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>System Uptime</span>
                <span>{formatUptime(stats.systemUptime)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '99.7%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Knowledge Base</span>
                <span>Healthy</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '95%' }} />
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Chat Volume</h3>
            <MessageSquare className="w-5 h-5 text-gray-500" />
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chatVolumeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="chats" fill="#3B82F6" name="Chat Requests" />
              <Bar dataKey="responses" fill="#10B981" name="Responses" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Response Time Trends</h3>
            <Clock className="w-5 h-5 text-gray-500" />
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="time" stroke="#3B82F6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Document Types</h3>
            <FileText className="w-5 h-5 text-gray-500" />
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={documentTypesData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {documentTypesData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-4">
            {documentTypesData.map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-gray-600">{item.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Knowledge Base Stats */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Knowledge Base Overview</h3>
          <div className="flex space-x-2">
            <button className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded-md hover:bg-gray-200">
              View Details
            </button>
            <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700">
              <Download className="w-4 h-4 inline mr-1" />
              Export
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {knowledgeBaseData.map((item, index) => (
            <div key={index} className="text-center">
              <div className="text-2xl font-bold text-gray-900">{formatNumber(item.value)}</div>
              <div className="text-sm text-gray-600 mt-1">{item.name}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {[
              { action: 'New document uploaded', user: 'john@example.com', time: '2 minutes ago' },
              { action: 'Chat session completed', user: 'sarah@example.com', time: '5 minutes ago' },
              { action: 'Knowledge graph updated', user: 'System', time: '12 minutes ago' },
              { action: 'User registered', user: 'mike@example.com', time: '1 hour ago' }
            ].map((activity, index) => (
              <div key={index} className="flex items-center justify-between py-2">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{activity.action}</p>
                    <p className="text-xs text-gray-500">{activity.user}</p>
                  </div>
                </div>
                <span className="text-xs text-gray-500">{activity.time}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}