<template>
    <div class="editor-layout">
        <!-- Top Navigation Bar -->
        <div class="top-nav">
            <div class="left-controls">
                <button class="home-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                        <polyline points="9 22 9 12 15 12 15 22" />
                    </svg>
                </button>
                <div class="document-info">
                    <span class="doc-title">Untitled</span>
                    <span class="doc-size">1536 × 1536</span>
                    <span class="doc-size-toggle">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m6 9 6 6 6-6" />
                        </svg>
                    </span>
                </div>
            </div>

            <div class="center-controls">
                <button class="tool-button play-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="8" x2="12" y2="16" />
                        <line x1="8" y1="12" x2="16" y2="12" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20 5H9l-7 7 7 7h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Z" />
                        <line x1="18" y1="9" x2="12" y2="15" />
                        <line x1="12" y1="9" x2="18" y2="15" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="14 15 9 20 4 15" />
                        <path d="M20 4h-7a4 4 0 1 0 0 8h7" />
                        <path d="M9 12h7a4 4 0 1 1 0 8H9" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M15 3h6v6" />
                        <path d="M10 14 21 3" />
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14 2 14 8 20 8" />
                    </svg>
                </button>
            </div>

            <div class="right-controls">
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M1 4v6h6" />
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                    </svg>
                </button>
                <button class="tool-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 4v6h-6" />
                        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                    </svg>
                </button>
                <div class="zoom-control">
                    <span>34%</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="m6 9 6 6 6-6" />
                    </svg>
                </div>
                <div class="credits">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M16 8h-6.5a2.5 2.5 0 0 0 0 5h3a2.5 2.5 0 0 1 0 5H6" />
                        <path d="M12 18v2m0-16v2" />
                    </svg>
                    <span>50</span>
                </div>
                <button class="export-button">Export</button>
                <button class="trial-button">Start FREE trial</button>
                <button class="help-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                        <path d="M12 17h.01" />
                    </svg>
                </button>
            </div>
        </div>

        <!-- Main Editor Area -->
        <div class="editor-container">
            <!-- Left Sidebar -->
            <aside class="left-sidebar">
          

                <!-- Tools List -->
                <div class="sidebar-tools">
                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Upload' }"
                        @click="setActiveTool('Upload')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <span>Upload</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Assets' }"
                        @click="setActiveTool('Assets')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="8" cy="11" r="5" />
                            <path d="M2.08 17.29A8 8 0 0 0 8 19a8 8 0 0 0 5.92-1.71" />
                            <path d="M19.85 13.88A8 8 0 0 0 21 9.5v-1" />
                            <path d="M19 7V5a2 2 0 1 0-4 0v2" />
                            <path d="M19 5a2 2 0 1 0 4 0a2 2 0 0 0-4 0Z" />
                            <path d="M22 12a10 10 0 0 1-4.5 8.5c-1.87 1.23-3.13 1.23-5 0A10 10 0 0 1 8 12" />
                        </svg>
                        <span>Assets</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Text to Image' }"
                        @click="setActiveTool('Text to Image')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M4 7V4h16v3" />
                            <path d="M9 20h6" />
                            <path d="M12 4v16" />
                        </svg>
                        <span>Txt2Img</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Image to Image' }"
                        @click="setActiveTool('Image to Image')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect width="18" height="18" x="3" y="3" rx="2" />
                            <circle cx="9" cy="9" r="2" />
                            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                        </svg>
                        <span>Img2Img</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Character' }"
                        @click="setActiveTool('Character')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
                            <circle cx="12" cy="7" r="4" />
                        </svg>
                        <span>Character</span>
                        <div class="hot-badge">HOT</div>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'AI Video' }"
                        @click="setActiveTool('AI Video')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m22 8-6 4 6 4V8Z" />
                            <rect width="14" height="12" x="2" y="6" rx="2" ry="2" />
                        </svg>
                        <span>AI Video</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Lip Sync' }"
                        @click="setActiveTool('Lip Sync')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17 9.2c5.7 1.2 5.7 11.4 0 12.6" />
                            <path
                                d="M14 7.2c3.6 1.2 3.6 13.4 0 14.6M7 8h2a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2Z" />
                        </svg>
                        <span>Lip Sync</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'AI Editor' }"
                        @click="setActiveTool('AI Editor')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon
                                points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                        </svg>
                        <span>AI Editor</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Face Kit' }"
                        @click="setActiveTool('Face Kit')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <circle cx="8" cy="10" r="1" />
                            <circle cx="16" cy="10" r="1" />
                            <path d="M8 15h.01" />
                            <path d="M16 15h.01" />
                            <path d="M9 10a4 4 0 0 1 6 0" />
                        </svg>
                        <span>Face Kit</span>
                    </button>

                    <button class="sidebar-tool-button" :class="{ active: activeTool === 'Enhance' }"
                        @click="setActiveTool('Enhance')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m2 12 5.25 5 2-2-5.3-5.3a1 1 0 0 0-1.4 0l-.55.6a1 1 0 0 0 0 1.7Z" />
                            <path
                                d="m15 12-8.53-8.51a1.5 1.5 0 0 0-1.56-.36l-2 .67a1.5 1.5 0 0 0-.91 1.95l.67 2c.1.28.26.53.47.71L12 17" />
                            <path d="m15 12 6.49 6.48a1 1 0 0 1-.93 1.52h-1.12a1 1 0 0 1-.7-.29L13.5 14.5" />
                        </svg>
                        <span>Enhance</span>
                    </button>

                    <button class="sidebar-tool-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect width="7" height="7" x="3" y="3" rx="1" />
                            <rect width="7" height="7" x="14" y="3" rx="1" />
                            <rect width="7" height="7" x="14" y="14" rx="1" />
                            <rect width="7" height="7" x="3" y="14" rx="1" />
                        </svg>
                        <span>Background</span>
                    </button>
                </div>

                <!-- Expand Button -->
                <div class="expand-sidebar">
                    <button class="expand-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m6 9 6 6 6-6" />
                        </svg>
                    </button>
                </div>
            </aside>

            <!-- Main Canvas Area -->
            <div class="canvas-area">
                <!-- Tool Bar -->
                <div class="tool-bar">
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10" stroke-dasharray="2" />
                        </svg>
                    </button>
                    <div class="tool-separator">|</div>
                    <button class="tool-bar-button">AI Eraser</button>
                    <button class="tool-bar-button">Hand Repair</button>
                    <button class="tool-bar-button">Expression</button>
                    <button class="tool-bar-button">BG Remove</button>
                    <div class="tool-separator">|</div>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5l6.74-6.76z" />
                            <line x1="16" y1="8" x2="2" y2="22" />
                            <line x1="17.5" y1="15" x2="9" y2="15" />
                        </svg>
                    </button>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 19V5" />
                            <path d="M5 12h14" />
                            <path d="M12 5v14" />
                        </svg>
                    </button>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                        </svg>
                    </button>
                    <div class="tool-separator">|</div>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M3 10a7 7 0 1 0 14 0 7 7 0 1 0-14 0" />
                            <path d="m21 21-4.35-4.35" />
                        </svg>
                    </button>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path
                                d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                        </svg>
                    </button>
                    <button class="tool-bar-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="3 11 22 2 13 21 11 13 3 11" />
                        </svg>
                    </button>
                    <button class="tool-bar-button" @click="openUploadWindow">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </button>
                </div>

                <!-- Canvas Area - Shows image1 by default (when no tool is selected) -->
                <div class="canvas-content">
                    <img v-if="!activeTool" src="https://placeholder.pics/svg/800x600/333333/FFFFFF/image1"
                        alt="Default Canvas State" class="default-image">

                    <div v-else-if="showUploadZone" class="canvas-drop-area">
                        <div class="drop-content">
                            <div class="drop-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24"
                                    fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                    stroke-linejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="17 8 12 3 7 8" />
                                    <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                            </div>
                            <h2 class="drop-message">CLICK, DRAG or PASTE here to upload assets</h2>
                            <div class="upload-options">
                                <input type="file" id="image-upload" accept="image/*" class="file-input-hidden"
                                    @change="handleFileUpload" />
                                <label for="image-upload" class="upload-button image-button">Image</label>
                                <span class="file-types">.JPG, .PNG, .WEBP or .PSD</span>
                            </div>
                            <div class="upload-options">
                                <input type="file" id="model-upload" accept=".glb,.gltf,.obj"
                                    class="file-input-hidden" />
                                <label for="model-upload" class="upload-button model-button">3D Model</label>
                                <span class="file-types">.GLB, .GLTF, .OBJ, .3DS, .DAE, .KMZ, .FBX or .STL</span>
                            </div>
                            <div class="watermark">dzine</div>
                        </div>
                    </div>

                    <div v-else-if="currentImage" class="canvas-image-container">
                        <img :src="currentImage" alt="Canvas Image" class="canvas-image">
                    </div>
                </div>
            </div>

            <!-- Right Sidebar -->
            <aside class="right-sidebar">
                <div class="sidebar-panel results-panel">
                    <div class="panel-header">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
                            <path
                                d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
                        </svg>
                        <h3>Results</h3>
                    </div>

                    <div class="no-results">
                        <div class="result-icon">
                            <img src="https://placeholder.pics/svg/100x100/DEDEDE/555555-DEDEDE/no-results"
                                alt="No Results">
                        </div>
                        <h4>No Results Available</h4>
                    </div>
                </div>

                <div class="sidebar-panel layers-panel">
                    <div class="panel-header">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m2 17 10 5 10-5" />
                            <path d="m2 12 10 5 10-5" />
                            <path d="m2 7 10 5 10-5" />
                            <path d="M12 2 2 7l10 5 10-5-10-5Z" />
                        </svg>
                        <h3>Layers</h3>
                        <button class="close-panel">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                stroke-linejoin="round">
                                <path d="M18 6 6 18" />
                                <path d="m6 6 12 12" />
                            </svg>
                        </button>
                    </div>

                    <div class="layer-list">
                        <div class="layer-item">
                            <div class="layer-thumbnail">
                                <img src="https://placeholder.pics/svg/50x50/DEDEDE/555555-DEDEDE/bg"
                                    alt="Background Layer">
                            </div>
                            <div class="layer-info">
                                <h4>Background</h4>
                                <span>No Fill</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Dzine Assistant - Now collapsed by default, expands on click -->
                <div class="assistant-panel" :class="{ 'expanded': assistantExpanded }">
                    <div class="assistant-header" @click="toggleAssistant">
                        <span>Dzine Assistant</span>
                        <button class="minimize-button">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                                fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                stroke-linejoin="round">
                                <path v-if="assistantExpanded" d="M18 15 12 9l-6 6" />
                                <path v-else d="M6 9l6 6 6-6" />
                            </svg>
                        </button>
                    </div>

                    <div class="assistant-content" v-if="assistantExpanded">
                        <div class="assistant-message">
                            <h3>Hi there 👋</h3>
                            <p>How can we help?</p>
                        </div>

                        <div class="assistant-options">
                            <div class="assistant-option">
                                <div class="option-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                                        fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                        stroke-linejoin="round">
                                        <circle cx="12" cy="5" r="1" />
                                        <path d="m9 20 3-6 3 6" />
                                        <path d="m6 8 6 2 6-2" />
                                        <path d="M12 10v4" />
                                    </svg>
                                </div>
                                <div class="option-content">
                                    <h4>Join Discord</h4>
                                    <p>Join a vibrant community and interact with Dzine experts.</p>
                                </div>
                            </div>

                            <div class="assistant-option">
                                <div class="option-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                                        fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                        stroke-linejoin="round">
                                        <path
                                            d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
                                        <path d="M9 18h6" />
                                        <path d="M10 22h4" />
                                    </svg>
                                </div>
                                <div class="option-content">
                                    <h4>Contact Support</h4>
                                    <p>Contact our team directly with any feedback you may have!</p>
                                </div>
                            </div>

                            <div class="assistant-option">
                                <div class="option-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
                                        fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
                                        stroke-linejoin="round">
                                        <circle cx="12" cy="12" r="10" />
                                        <path d="M12 8v4l3 3" />
                                    </svg>
                                </div>
                                <div class="option-content">
                                    <h4>Prompt Assistant</h4>
                                    <p>Start chatting with our Dzine prompt assistant now!</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>
        </div>
    </div>
</template>
    
    <script setup>
    import { ref, computed } from 'vue';

    // Tool state and selection
    const activeTool = ref(null); // No tool selected by default to show image1
    const textToImagePrompt = ref('A transparent perfume bottle elegantly sits in the center of a wooden table, reflecting the soft lighting of the room. On either side, blossoming flowers with delicate petals of various radiant hues curate a romantic ambiance. The background is subtly blurred, coaxing the observer\'s focus towards the central display.');

    // Current image state (for displaying uploaded or generated images)
    const currentImage = ref(null);

    // Assistant state
    const assistantExpanded = ref(false);

    // Upload functionality
    const hasRecentUploads = ref(false);
    const showUploadZone = computed(() => {
        return ["Upload", "Text to Image", "Image to Image"].includes(activeTool.value);
    });

    // Set active tool and handle tool-specific behavior
    const setActiveTool = (tool) => {
        activeTool.value = tool;

        // Reset current image when switching tools
        if (currentImage.value && tool !== "Upload") {
            currentImage.value = null;
        }
    };

    // Close the active tool panel and return to default state (image1)
    const closeTool = () => {
        activeTool.value = null;
        currentImage.value = null;
    };

    // Toggle assistant panel expansion
    const toggleAssistant = () => {
        assistantExpanded.value = !assistantExpanded.value;
    };

    // Handle file upload
    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImage.value = e.target.result;
                hasRecentUploads.value = true;
            };
            reader.readAsDataURL(file);
        }
    };

    // Open upload window from toolbar
    const openUploadWindow = () => {
        setActiveTool('Upload');
    };
</script>
    
    <style scoped>

    /* Main Layout */
    .editor-layout {
        display: flex;
        flex-direction: column;
        height: 100vh;
        background-color: #0e0e0e;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
    }

    /* Top Navigation */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 56px;
        padding: 0 1rem;
        background-color: #1a1a1a;
        border-bottom: 1px solid #222;
    }

    .left-controls,
    .center-controls,
    .right-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .document-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #ccc;
        font-size: 0.9rem;
    }

    .doc-title {
        font-weight: 500;
    }

    .doc-size,
    .doc-size-toggle {
        color: #888;
    }

    .tool-button {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .tool-button:hover {
        background-color: #333;
        color: #fff;
    }

    .play-button {
        background-color: #D4AF37;
        color: #000;
    }

    .play-button:hover {
        background-color: #e6c34a;
        color: #000;
    }

    .zoom-control {
        display: flex;
        align-items: center;
        color: #ccc;
        font-size: 0.9rem;
        gap: 0.25rem;
        padding: 0 0.5rem;
    }

    .credits {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #ccc;
        font-size: 0.9rem;
    }

    .export-button {
        padding: 0.5rem 1rem;
        background-color: transparent;
        border: 1px solid #444;
        border-radius: 6px;
        color: #ccc;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .export-button:hover {
        background-color: #333;
    }

    .trial-button {
        padding: 0.5rem 1rem;
        background-color: #D4AF37;
        border: none;
        border-radius: 6px;
        color: #000;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .trial-button:hover {
        background-color: #e6c34a;
    }

    .help-button {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
    }

    /* Editor Container */
    .editor-container {
        display: flex;
        flex: 1;
        overflow: hidden;
    }

    /* Left Sidebar */
    .left-sidebar {
        width: 320px;
        height: auto;
        display: flex;
        flex-direction: column;
        background-color: #1a1a1a;
        border-right: 1px solid #222;
        overflow: hidden;
        /* Hide overflow from container itself */
    }

    .tool-tabs {
        flex: 1;
        overflow-y: auto;
        /* Add scrolling to this element */
        padding: 1rem;
    }

    .active-tool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .active-tool-header h3 {
        font-size: 1.25rem;
        font-weight: 500;
        margin: 0;
    }

    .close-panel-btn {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
        border-radius: 6px;
    }

    .close-panel-btn:hover {
        background-color: #333;
    }

    .tool-options-panel {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .model-selector {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #222;
        padding: 0.5rem;
        border-radius: 6px;
    }

    .model-preview {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .model-icon {
        width: 32px;
        height: 32px;
        border-radius: 4px;
    }

    .model-toggle,
    .add-model {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
        border-radius: 6px;
    }

    .prompt-area,
    .prompt-placeholder {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .prompt-input {
        min-height: 150px;
        background-color: #222;
        border: none;
        border-radius: 6px;
        color: #fff;
        padding: 0.75rem;
        font-family: inherit;
        resize: vertical;
    }

    .prompt-counter {
        align-self: flex-end;
        font-size: 0.8rem;
        color: #888;
    }

    .describe-counter {
        align-self: flex-end;
        font-size: 0.8rem;
        color: #888;
    }

    .describe-canvas {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        color: #D4AF37;
        cursor: pointer;
        padding: 0.5rem;
    }

    .option-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
    }

    .option-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
    }

    .toggle-switch {
        width: 36px;
        height: 20px;
        background-color: #333;
        border-radius: 10px;
        position: relative;
        cursor: pointer;
    }

    .badge {
        font-size: 0.7rem;
        font-weight: 600;
        background-color: #D4AF37;
        color: #000;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
    }

    .help-icon {
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #888;
        cursor: pointer;
    }

    .aspect-value {
        font-size: 0.9rem;
        color: #ccc;
    }

    .aspect-buttons {
        display: flex;
        gap: 0.5rem;
    }

    .aspect-button {
        flex: 1;
        padding: 0.5rem 0;
        background-color: #333;
        border: none;
        border-radius: 4px;
        color: #ccc;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .aspect-button.active {
        background-color: #444;
        color: #fff;
    }

    .generation-modes {
        display: flex;
        gap: 0.5rem;
    }

    .mode-button {
        flex: 1;
        padding: 0.75rem 0;
        background-color: #333;
        border: none;
        border-radius: 4px;
        color: #ccc;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .mode-button.active {
        background-color: #444;
        color: #fff;
    }

    .slider-control {
        margin-top: -0.5rem;
    }

    .slider-track {
        height: 4px;
        background-color: #333;
        border-radius: 2px;
        position: relative;
        margin: 0.5rem 0;
    }

    .slider-fill {
        position: absolute;
        height: 100%;
        background-color: #D4AF37;
        border-radius: 2px;
    }

    .slider-handle {
        position: absolute;
        width: 16px;
        height: 16px;
        background-color: #fff;
        border-radius: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        cursor: pointer;
    }

    .slider-labels {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #888;
    }

    .layers-warning {
        color: #888;
        font-size: 0.9rem;
        margin-top: 1rem;
    }

    .generate-button {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background-color: #D4AF37;
        border: none;
        border-radius: 6px;
        color: #000;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1rem;
    }

    .generate-label {
        font-size: 0.8rem;
        font-weight: normal;
        background-color: rgba(0, 0, 0, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }

    .credit-amount {
        background-color: #fff;
        color: #000;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
    }

    .action-button {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem;
        background-color: #222;
        border: 1px solid #444;
        border-radius: 6px;
        color: #fff;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .build-button {
        background-color: #D4AF37;
        color: #000;
        border: none;
    }

    .character-options {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .option-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background-color: #222;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
    }

    .option-card:hover {
        background-color: #333;
    }

    .option-info h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
    }

    .option-info p {
        margin: 0;
        font-size: 0.8rem;
        color: #888;
    }

    .emoji-icon {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        font-size: 1.2rem;
    }

    /* Upload Container Styles */
    .upload-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .upload-title {
        font-size: 1.2rem;
        font-weight: 500;
    }

    .upload-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background-color: #222;
        border: 2px dashed #444;
        border-radius: 8px;
        gap: 1rem;
    }

    .upload-icon {
        width: 70px;
        height: 70px;
        background-color: #333;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #D4AF37;
    }

    .upload-text {
        font-size: 1.1rem;
        font-weight: 500;
    }

    .upload-formats {
        color: #888;
        font-size: 0.9rem;
    }

    .upload-button-label {
        padding: 0.75rem 1.5rem;
        background-color: #D4AF37;
        color: #000;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1rem;
    }

    .upload-input {
        display: none;
    }

    .recent-uploads {
        margin-top: 1rem;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }

    .uploads-empty {
        color: #888;
        font-size: 0.9rem;
        text-align: center;
        padding: 1rem;
        background-color: #222;
        border-radius: 6px;
    }

    .upload-thumbnails {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .upload-thumbnail {
        width: 80px;
        height: 80px;
        background-color: #222;
        border-radius: 4px;
        overflow: hidden;
    }

    .upload-thumbnail img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Assets Panel */
    .assets-search {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .search-input-container {
        flex: 1;
        position: relative;
    }

    .search-icon {
        position: absolute;
        left: 0.75rem;
        top: 50%;
        transform: translateY(-50%);
        color: #888;
    }

    .search-input {
        width: 100%;
        padding: 0.75rem 0.75rem 0.75rem 2.5rem;
        background-color: #222;
        border: none;
        border-radius: 6px;
        color: #fff;
    }

    .filter-button {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background-color: #222;
        border: none;
        border-radius: 6px;
        color: #ccc;
        cursor: pointer;
    }

    .assets-categories {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .category-button {
        padding: 0.5rem 1rem;
        background-color: #222;
        border-radius: 6px;
        white-space: nowrap;
        cursor: pointer;
    }

    .category-button.active {
        background-color: #333;
        color: #fff;
    }

    .assets-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
    }

    .asset-card {
        aspect-ratio: 1/1;
        background-color: #222;
        border-radius: 6px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .asset-card:hover {
        transform: translateY(-2px);
    }

    .asset-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Sidebar Tools */
    .sidebar-tools {
        display: grid;
        grid-template-columns: repeat(1, 1fr);
        gap: 0.5rem;
        padding: 0.5rem;
        border-top: 1px solid #222;
        max-height: 100%;
        /* Limit height */
        overflow-y: auto;
        /* Make scrollable */
    }

    .sidebar-tool-button {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.25rem;
        padding: 0.75rem;
        background-color: transparent;
        border: none;
        border-radius: 6px;
        color: #ccc;
        font-size: 0.8rem;
        cursor: pointer;
        position: relative;
        transition: all 0.2s ease;
    }

    .sidebar-tool-button:hover,
    .sidebar-tool-button.active {
        background-color: #222;
        color: #fff;
    }

    .hot-badge {
        position: absolute;
        top: 0;
        right: 25%;
        background-color: #f44336;
        color: #fff;
        font-size: 0.6rem;
        font-weight: 600;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
    }

    .expand-sidebar {
        display: flex;
        justify-content: center;
        padding: 0.5rem;
        border-top: 1px solid #222;
    }

    .expand-button {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        border: none;
        color: #888;
        cursor: pointer;
        border-radius: 6px;
    }

    /* Canvas Area */
    .canvas-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background-color: #121212;
        position: relative;
    }

    .tool-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background-color: #1a1a1a;
        border-bottom: 1px solid #222;
        overflow-x: auto;
        /* Allow horizontal scrolling */
    }

    .tool-bar-button {
        padding: 0.5rem 0.75rem;
        background-color: transparent;
        border: none;
        border-radius: 6px;
        color: #ccc;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
    }

    .tool-bar-button:hover {
        background-color: #333;
        color: #fff;
    }

    .tool-separator {
        color: #444;
        padding: 0 0.25rem;
    }

    .canvas-content {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        overflow: hidden;
        position: relative;
    }

    .default-image {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    .canvas-image-container {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .canvas-image {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    .canvas-drop-area {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .drop-content {
        max-width: 600px;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
    }

    .drop-icon {
        width: 80px;
        height: 80px;
        background-color: #333;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ccc;
        margin-bottom: 2rem;
    }

    .drop-message {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 2rem;
        color: #ccc;
    }

    .upload-options {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .upload-button {
        padding: 0.5rem 0.75rem;
        border: none;
        border-radius: 6px;
        font-size: 0.9rem;
        cursor: pointer;
    }

    .image-button {
        background-color: #2196F3;
        color: #fff;
    }

    .model-button {
        background-color: #D4AF37;
        color: #000;
    }

    .file-types {
        font-size: 0.9rem;
        color: #888;
    }

    .file-input-hidden {
        display: none;
    }

    .watermark {
        position: absolute;
        bottom: -3rem;
        right: 0;
        font-size: 1.5rem;
        opacity: 0.2;
        color: #ccc;
    }

    /* Right Sidebar */
    .right-sidebar {
        width: 320px;
        height: 100%;
        display: flex;
        flex-direction: column;
        background-color: #1a1a1a;
        border-left: 1px solid #222;
    }

    .sidebar-panel {
        flex: 1;
        overflow-y: auto;
    }

    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        border-bottom: 1px solid #222;
    }

    .panel-header h3 {
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        flex: 1;
    }

    .close-panel {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
        border-radius: 6px;
    }

    .close-panel:hover {
        background-color: #333;
    }

    .no-results {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 300px;
        color: #888;
        text-align: center;
    }

    .result-icon {
        margin-bottom: 1rem;
    }

    .layer-list {
        padding: 1rem;
    }

    .layer-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .layer-item:hover {
        background-color: #222;
    }

    .layer-thumbnail {
        width: 40px;
        height: 40px;
        border-radius: 4px;
        overflow: hidden;
        background-color: #333;
    }

    .layer-info h4 {
        margin: 0;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .layer-info span {
        font-size: 0.8rem;
        color: #888;
    }

    /* Assistant Panel - Modified to be collapsible */
    .assistant-panel {
        margin-top: auto;
        border-top: 1px solid #222;
    }

    .assistant-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        background-color: #000;
        color: #fff;
        cursor: pointer;
    }

    .minimize-button {
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: transparent;
        border: none;
        color: #ccc;
        cursor: pointer;
        border-radius: 4px;
    }

    .minimize-button:hover {
        background-color: #333;
    }

    .assistant-content {
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        background-color: #1a1a1a;
    }

    /* Only show assistant when expanded */
    .assistant-panel:not(.expanded) .assistant-content {
        display: none;
    }

    .assistant-message h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0 0 0.5rem;
    }

    .assistant-message p {
        margin: 0;
        color: #ccc;
    }

    .assistant-options {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .assistant-option {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        background-color: #222;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .assistant-option:hover {
        background-color: #333;
    }

    .option-icon {
        color: #D4AF37;
    }

    .option-content h4 {
        margin: 0 0 0.25rem;
        font-size: 1rem;
        font-weight: 500;
    }

    .option-content p {
        margin: 0;
        font-size: 0.8rem;
        color: #888;
    }

    /* Responsive Styles */
    @media (max-width: 1400px) {

        .right-sidebar,
        .left-sidebar {
            width: 280px;
        }
    }

    @media (max-width: 1200px) {
        .right-sidebar {
            display: none;
        }
    }

    @media (max-width: 992px) {
        .left-sidebar {
            width: 240px;
        }

        .tool-bar-button {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
    }

    @media (max-width: 768px) {
        .top-nav {
            padding: 0 0.5rem;
        }

        .center-controls,
        .left-controls .document-info {
            display: none;
        }

        .export-button {
            display: none;
        }

        .left-sidebar {
            position: absolute;
            left: 0;
            top: 56px;
            bottom: 0;
            z-index: 100;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }

        .left-sidebar.open {
            transform: translateX(0);
        }

        .tool-bar {
            overflow-x: auto;
            justify-content: flex-start;
        }
    }
</style>
