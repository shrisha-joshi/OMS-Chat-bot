/**
 * YouTubePlayer Component
 * Displays YouTube videos with fullscreen, playback controls, and responsive sizing.
 * Used in chat responses to show video media.
 */

'use client';

import React, { useState } from 'react';
import { Maximize2, Minimize2, Volume2, VolumeX } from 'lucide-react';

interface YouTubePlayerProps {
  videoId: string;
  title?: string;
  width?: number | string;
  height?: number | string;
  autoplay?: boolean;
  onClose?: () => void;
}

const YouTubePlayer: React.FC<YouTubePlayerProps> = ({
  videoId,
  title = 'YouTube Video',
  width = '100%',
  height = '400px',
  autoplay = false,
  onClose
}) => {
    const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState(false);
  const [containerRef, setContainerRef] = useState<HTMLDivElement | null>(null);

  const handleFullscreen = () => {
    if (!containerRef) return;

    const exitFullscreen = async () => {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
      } else if ((document as any).msExitFullscreen) {
        (document as any).msExitFullscreen();
      }
      setIsFullscreen(false);
    };

    const enterFullscreen = async () => {
      if (containerRef.requestFullscreen) {
        await containerRef.requestFullscreen();
      } else if ((containerRef as any).webkitRequestFullscreen) {
        (containerRef as any).webkitRequestFullscreen();
      } else if ((containerRef as any).msRequestFullscreen) {
        (containerRef as any).msRequestFullscreen();
      }
      setIsFullscreen(true);
    };

    if (document.fullscreenElement || isFullscreen) {
      void exitFullscreen();
    } else {
      void enterFullscreen();
    }
  };

  const embedUrl = `https://www.youtube.com/embed/${videoId}?${[
    autoplay ? 'autoplay=1' : 'autoplay=0',
    'controls=1',
    'modestbranding=1',
    'rel=0',
    isMuted ? 'mute=1' : 'mute=0'
  ].join('&')}`;

  return (
    <div
      ref={setContainerRef}
      className={`relative bg-black rounded-lg overflow-hidden ${
        isFullscreen ? 'fixed inset-0 z-[9999]' : ''
      }`}
      style={{
        width: isFullscreen ? '100vw' : width,
        height: isFullscreen ? '100vh' : height,
      }}
    >
      {/* Video Container */}
      <div className="relative w-full h-full">
        <iframe
          src={embedUrl}
          title={title}
          className="w-full h-full"
          style={{border: 0}}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          loading="lazy"
        />
      </div>

      {/* Controls Overlay */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/50 to-transparent p-4 flex justify-between items-center group hover:opacity-100 opacity-0 transition-opacity">
        <div className="flex-1">
          <h3 className="text-white text-sm font-semibold truncate">{title}</h3>
        </div>
        <div className="flex items-center gap-2">
          {/* Mute Button */}
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="p-2 rounded bg-black/50 hover:bg-black/75 text-white transition-colors"
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </button>

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

      {/* Bottom Controls */}
      {isFullscreen && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-4 flex justify-end gap-2">
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

export default YouTubePlayer;
