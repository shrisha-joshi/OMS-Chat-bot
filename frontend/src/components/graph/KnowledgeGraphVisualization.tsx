'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Search, Filter, ZoomIn, ZoomOut, RotateCcw, Share2, Download, Eye, EyeOff } from 'lucide-react'

interface Node {
  id: string
  label: string
  type: 'entity' | 'document' | 'concept'
  properties: Record<string, any>
  x?: number
  y?: number
  connections: number
}

interface Edge {
  id: string
  source: string
  target: string
  label: string
  type: 'mentions' | 'contains' | 'relates_to' | 'similar_to'
  weight: number
}

interface KnowledgeGraphData {
  nodes: Node[]
  edges: Edge[]
}

interface KnowledgeGraphVisualizationProps {
  data?: KnowledgeGraphData
  height?: number
  onNodeClick?: (node: Node) => void
  onEdgeClick?: (edge: Edge) => void
}

export function KnowledgeGraphVisualization({ 
  data, 
  height = 600, 
  onNodeClick, 
  onEdgeClick 
}: KnowledgeGraphVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedNodeType, setSelectedNodeType] = useState<string>('all')
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [showLabels, setShowLabels] = useState(true)

  // Sample data for demonstration
  const sampleData: KnowledgeGraphData = {
    nodes: [
      { id: '1', label: 'Document A.pdf', type: 'document', properties: { size: 1024, pages: 5 }, connections: 3 },
      { id: '2', label: 'Machine Learning', type: 'concept', properties: { category: 'AI' }, connections: 4 },
      { id: '3', label: 'Neural Networks', type: 'concept', properties: { category: 'AI' }, connections: 2 },
      { id: '4', label: 'John Smith', type: 'entity', properties: { role: 'Author' }, connections: 2 },
      { id: '5', label: 'Report B.docx', type: 'document', properties: { size: 2048, pages: 12 }, connections: 3 },
      { id: '6', label: 'Data Analysis', type: 'concept', properties: { category: 'Analytics' }, connections: 3 },
      { id: '7', label: 'Python', type: 'entity', properties: { type: 'Technology' }, connections: 4 },
      { id: '8', label: 'Research Paper', type: 'document', properties: { size: 4096, pages: 20 }, connections: 2 }
    ],
    edges: [
      { id: 'e1', source: '1', target: '2', label: 'mentions', type: 'mentions', weight: 0.8 },
      { id: 'e2', source: '2', target: '3', label: 'relates to', type: 'relates_to', weight: 0.9 },
      { id: 'e3', source: '1', target: '4', label: 'authored by', type: 'contains', weight: 1.0 },
      { id: 'e4', source: '5', target: '6', label: 'discusses', type: 'mentions', weight: 0.7 },
      { id: 'e5', source: '6', target: '7', label: 'uses', type: 'relates_to', weight: 0.6 },
      { id: 'e6', source: '2', target: '7', label: 'implemented in', type: 'relates_to', weight: 0.8 },
      { id: 'e7', source: '8', target: '2', label: 'studies', type: 'mentions', weight: 0.85 },
      { id: 'e8', source: '3', target: '7', label: 'coded in', type: 'relates_to', weight: 0.7 }
    ]
  }

  const graphData = data || sampleData

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'document': return '#3B82F6'
      case 'concept': return '#10B981'
      case 'entity': return '#F59E0B'
      default: return '#6B7280'
    }
  }

  const getEdgeColor = (type: string) => {
    switch (type) {
      case 'mentions': return '#8B5CF6'
      case 'contains': return '#EF4444'
      case 'relates_to': return '#06B6D4'
      case 'similar_to': return '#84CC16'
      default: return '#9CA3AF'
    }
  }

  const filteredNodes = graphData.nodes.filter(node => {
    const matchesSearch = node.label.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = selectedNodeType === 'all' || node.type === selectedNodeType
    return matchesSearch && matchesType
  })

  const filteredEdges = graphData.edges.filter(edge => {
    const sourceExists = filteredNodes.some(node => node.id === edge.source)
    const targetExists = filteredNodes.some(node => node.id === edge.target)
    return sourceExists && targetExists
  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
    canvas.style.width = `${canvas.offsetWidth}px`
    canvas.style.height = `${height}px`

    // Clear canvas
    ctx.clearRect(0, 0, canvas.offsetWidth, height)

    // Apply transformations
    ctx.save()
    ctx.translate(pan.x, pan.y)
    ctx.scale(zoom, zoom)

    // Position nodes in a circle for demonstration
    const centerX = canvas.offsetWidth / 2 / zoom
    const centerY = height / 2 / zoom
    const radius = Math.min(centerX, centerY) * 0.7

    filteredNodes.forEach((node, index) => {
      const angle = (index / filteredNodes.length) * 2 * Math.PI
      node.x = centerX + Math.cos(angle) * radius
      node.y = centerY + Math.sin(angle) * radius
    })

    // Draw edges
    filteredEdges.forEach(edge => {
      const sourceNode = filteredNodes.find(n => n.id === edge.source)
      const targetNode = filteredNodes.find(n => n.id === edge.target)
      
      if (sourceNode && targetNode && sourceNode.x && sourceNode.y && targetNode.x && targetNode.y) {
        ctx.beginPath()
        ctx.strokeStyle = getEdgeColor(edge.type)
        ctx.lineWidth = edge.weight * 2
        ctx.globalAlpha = 0.7
        ctx.moveTo(sourceNode.x, sourceNode.y)
        ctx.lineTo(targetNode.x, targetNode.y)
        ctx.stroke()

        // Draw edge label if zoomed in enough
        if (zoom > 0.8 && showLabels) {
          const midX = (sourceNode.x + targetNode.x) / 2
          const midY = (sourceNode.y + targetNode.y) / 2
          
          ctx.font = '12px Inter, sans-serif'
          ctx.fillStyle = '#4B5563'
          ctx.textAlign = 'center'
          ctx.globalAlpha = 1
          ctx.fillText(edge.label, midX, midY - 5)
        }
      }
    })

    // Draw nodes
    filteredNodes.forEach(node => {
      if (node.x && node.y) {
        const isSelected = selectedNode?.id === node.id
        const isHovered = hoveredNode?.id === node.id
        const nodeRadius = 20 + (node.connections * 2)

        // Node shadow
        if (isSelected || isHovered) {
          ctx.beginPath()
          ctx.arc(node.x, node.y, nodeRadius + 5, 0, 2 * Math.PI)
          ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
          ctx.fill()
        }

        // Node circle
        ctx.beginPath()
        ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI)
        ctx.fillStyle = getNodeColor(node.type)
        ctx.globalAlpha = isSelected || isHovered ? 1 : 0.8
        ctx.fill()

        // Node border
        if (isSelected) {
          ctx.strokeStyle = '#1F2937'
          ctx.lineWidth = 3
          ctx.stroke()
        }

        // Node label
        if (showLabels) {
          ctx.font = `${isSelected || isHovered ? '14px' : '12px'} Inter, sans-serif`
          ctx.fillStyle = '#1F2937'
          ctx.textAlign = 'center'
          ctx.globalAlpha = 1
          
          const labelY = node.y + nodeRadius + 15
          const maxWidth = 120
          const words = node.label.split(' ')
          let line = ''
          let y = labelY

          for (let n = 0; n < words.length; n++) {
            const testLine = line + words[n] + ' '
            const metrics = ctx.measureText(testLine)
            const testWidth = metrics.width
            
            if (testWidth > maxWidth && n > 0) {
              ctx.fillText(line, node.x, y)
              line = words[n] + ' '
              y += 16
            } else {
              line = testLine
            }
          }
          ctx.fillText(line, node.x, y)
        }
      }
    })

    ctx.restore()
  }, [filteredNodes, filteredEdges, zoom, pan, selectedNode, hoveredNode, showLabels, height])

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left - pan.x) / zoom
    const y = (event.clientY - rect.top - pan.y) / zoom

    // Find clicked node
    const clickedNode = filteredNodes.find(node => {
      if (!node.x || !node.y) return false
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
      return distance <= 20 + (node.connections * 2)
    })

    if (clickedNode) {
      setSelectedNode(clickedNode)
      onNodeClick?.(clickedNode)
    } else {
      setSelectedNode(null)
    }
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 0.3))
  const handleReset = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setSelectedNode(null)
  }

  const nodeTypes = ['all', ...Array.from(new Set(graphData.nodes.map(n => n.type)))]

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Knowledge Graph</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowLabels(!showLabels)}
              className={`p-2 rounded-md ${showLabels ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}
              title={showLabels ? 'Hide labels' : 'Show labels'}
            >
              {showLabels ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
            <button className="p-2 bg-gray-100 text-gray-500 rounded-md hover:bg-gray-200" title="Share">
              <Share2 className="w-4 h-4" />
            </button>
            <button className="p-2 bg-gray-100 text-gray-500 rounded-md hover:bg-gray-200" title="Download">
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <select
              value={selectedNodeType}
              onChange={(e) => setSelectedNodeType(e.target.value)}
              className="pl-10 pr-8 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            >
              {nodeTypes.map(type => (
                <option key={type} value={type}>
                  {type === 'all' ? 'All Types' : type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center space-x-1 border border-gray-300 rounded-md">
            <button
              onClick={handleZoomOut}
              className="p-2 hover:bg-gray-100"
              title="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="px-2 py-1 text-sm font-mono border-l border-r border-gray-300 min-w-[60px] text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-2 hover:bg-gray-100"
              title="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              className="p-2 hover:bg-gray-100 border-l border-gray-300"
              title="Reset view"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="w-full cursor-crosshair"
          style={{ height: `${height}px` }}
        />

        {/* Stats */}
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-90 backdrop-blur-sm rounded-md p-3 text-sm">
          <div className="flex items-center space-x-4 text-gray-600">
            <span>{filteredNodes.length} nodes</span>
            <span>{filteredEdges.length} edges</span>
            <span>Zoom: {Math.round(zoom * 100)}%</span>
          </div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-white bg-opacity-90 backdrop-blur-sm rounded-md p-3">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Node Types</h4>
          <div className="space-y-1 text-xs">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-gray-600">Documents</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-gray-600">Concepts</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span className="text-gray-600">Entities</span>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <h4 className="font-medium text-gray-900 mb-2">{selectedNode.label}</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Type:</span>
              <span className="ml-2 text-gray-900">{selectedNode.type}</span>
            </div>
            <div>
              <span className="text-gray-500">Connections:</span>
              <span className="ml-2 text-gray-900">{selectedNode.connections}</span>
            </div>
          </div>
          {Object.keys(selectedNode.properties).length > 0 && (
            <div className="mt-2">
              <span className="text-gray-500 text-sm">Properties:</span>
              <div className="mt-1 text-sm text-gray-900">
                {Object.entries(selectedNode.properties).map(([key, value]) => (
                  <div key={key} className="inline-block mr-3">
                    <span className="font-medium">{key}:</span> {String(value)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}