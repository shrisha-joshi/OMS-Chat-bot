/**
 * ImageViewer Component
 * Displays images with zoom, fullscreen, and lightbox capabilities.
 * Used in chat responses to show image media.
 */

'use client';

import React, { useState, useRef } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Minimize2, ChevronLeft, ChevronRight } from 'lucide-react';

interface ImageViewerProps {
  src: string;
  alt?: string;
  title?: string;
  width?: number | string;
  height?: number | string;
  maxWidth?: string;
  onClose?: () => void;
  gallery?: string[];  // For gallery/carousel mode
}

const ImageViewer: React.FC<ImageViewerProps> = ({
  src,
  alt = 'Image',
  title = 'Image',
  width = '100%',
  height = 'auto',
  maxWidth = '600px',
  onClose,
  gallery = []
}) => {
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const currentSrc = gallery.length > 0 ? gallery[currentImageIndex] : src;
  const hasGallery = gallery.length > 1;

  const handleZoom = (direction: 'in' | 'out') => {
    setZoom((prev) => {
      const newZoom = direction === 'in' ? prev + 0.2 : Math.max(1, prev - 0.2);
      return Math.min(3, newZoom);  // Max 3x zoom
    });
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPanPosition({ x: 0, y: 0 });
  };

  const handleFullscreen = () => {
    if (!containerRef.current) return;

    if (isFullscreen) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
      setIsFullscreen(false);
    } else if (containerRef.current.requestFullscreen) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } else if ((containerRef.current as any).webkitRequestFullscreen) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (containerRef.current as any).webkitRequestFullscreen();
      setIsFullscreen(true);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } else if ((containerRef.current as any).msRequestFullscreen) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (containerRef.current as any).msRequestFullscreen();
      setIsFullscreen(true);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - panPosition.x, y: e.clientY - panPosition.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setPanPosition({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y
    });
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handlePrevImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + gallery.length) % gallery.length);
    handleResetZoom();
  };

  const handleNextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % gallery.length);
    handleResetZoom();
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const direction = e.deltaY < 0 ? 'in' : 'out';
    handleZoom(direction);
  };

  return (
    <div
      ref={containerRef}
      className={`relative bg-gray-900 rounded-lg overflow-hidden select-none ${
        isFullscreen ? 'fixed inset-0 z-[9999]' : ''
      }`}
      style={{
        width: isFullscreen ? '100vw' : width,
        height: isFullscreen ? '100vh' : height,
        maxWidth: isFullscreen ? undefined : maxWidth
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      <div
        className="relative w-full h-full flex items-center justify-center overflow-hidden cursor-move"
        role="img"
        aria-label={alt}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imageRef}
          src={currentSrc}
          alt={alt}
          className="max-w-full max-h-full object-contain transition-transform duration-200"
          style={{
            transform: `scale(${zoom}) translate(${panPosition.x / zoom}px, ${panPosition.y / zoom}px)`,
            cursor: zoom > 1 ? 'grab' : 'default'
          }}
          draggable={false}
        />
      </div>

      {/* Top Controls */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/50 to-transparent p-4 flex justify-between items-center group hover:opacity-100 opacity-0 transition-opacity">
        <div className="flex-1">
          <h3 className="text-white text-sm font-semibold truncate">{title}</h3>
          {hasGallery && (
            <p className="text-white/70 text-xs">
              {currentImageIndex + 1} / {gallery.length}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Zoom Controls */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => handleZoom('out')}
              className="p-2 rounded bg-black/50 hover:bg-black/75 text-white transition-colors disabled:opacity-50"
              disabled={zoom <= 1}
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-white text-xs px-2 min-w-[40px] text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => handleZoom('in')}
              className="p-2 rounded bg-black/50 hover:bg-black/75 text-white transition-colors disabled:opacity-50"
              disabled={zoom >= 3}
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            {zoom > 1 && (
              <button
                onClick={handleResetZoom}
                className="px-2 py-1 rounded bg-black/50 hover:bg-black/75 text-white text-xs transition-colors"
                title="Reset Zoom"
              >
                Reset
              </button>
            )}
          </div>

          {/* Fullscreen Button */}
          <button
            onClick={handleFullscreen}
            className="p-2 rounded bg-black/50 hover:bg-black/75 text-white transition-colors"
            title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>

          {/* Close Button */}
          {onClose && !isFullscreen && (
            <button
              onClick={onClose}
              className="p-2 rounded bg-black/50 hover:bg-black/75 text-white transition-colors"
              title="Close"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Gallery Navigation */}
      {hasGallery && (
        <>
          {/* Left Arrow */}
          <button
            onClick={handlePrevImage}
            className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-black/50 hover:bg-black/75 text-white transition-colors opacity-0 group-hover:opacity-100 hover:opacity-100 z-10"
            title="Previous Image"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>

          {/* Right Arrow */}
          <button
            onClick={handleNextImage}
            className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-black/50 hover:bg-black/75 text-white transition-colors opacity-0 group-hover:opacity-100 hover:opacity-100 z-10"
            title="Next Image"
          >
            <ChevronRight className="w-6 h-6" />
          </button>

          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
            {gallery.map((imgSrc, index) => (
              <button
                key={imgSrc}
                onClick={() => {
                  setCurrentImageIndex(index);
                  handleResetZoom();
                }}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentImageIndex
                    ? 'bg-white'
                    : 'bg-white/50 hover:bg-white/75'
                }`}
                title={`Go to image ${index + 1}`}
              />
            ))}
          </div>
        </>
      )}

      {/* Fullscreen Bottom Controls */}
      {isFullscreen && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-4 flex justify-between items-center">
          <div className="text-white">
            <p className="text-sm font-semibold">{title}</p>
            <p className="text-xs text-white/70">{alt}</p>
          </div>
          <button
            onClick={handleFullscreen}
            className="px-4 py-2 rounded bg-red-600 hover:bg-red-700 text-white transition-colors"
          >
            Exit Fullscreen
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
