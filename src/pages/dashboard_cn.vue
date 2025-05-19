<template>
    <div class="dashboard-layout">
      <!-- Left Sidebar -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <a href="/" class="logo">
            <img src="https://placeholder.pics/svg/120x40/333333/FFFFFF-333333/dzine" alt="Dzine Logo" class="logo-img">
            <span class="beta-tag">BETA</span>
          </a>
        </div>
  
        <div class="sidebar-content">
          <!-- User Profile -->
          <div class="user-profile">
            <div class="avatar">
              <img src="https://placeholder.pics/svg/50x50/CCCCCC/666666-CCCCCC/user" alt="User Avatar">
            </div>
            <div class="user-info">
              <div class="user-id">用户名<span class="chevron">›</span></div>
              <div class="user-plan">免费用户</div>
            </div>
          </div>
  
          <!-- Credits -->
          <div class="credits-section">
            <div class="credits-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M16 8h-6.5a2.5 2.5 0 0 0 0 5h3a2.5 2.5 0 0 1 0 5H6"/><path d="M12 18v2m0-16v2"/></svg>
            </div>
            <div class="credits-label">点数</div>
            <div class="credits-value">50</div>
          </div>
  
          <!-- Premium Promo -->
          <div class="premium-promo">
            <div class="crown-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 4 3 12h14l3-12-6 7-4-7-4 7-6-7zm3 16h14"/></svg>
            </div>
            <div class="premium-text">
              <div class="premium-title">高级版</div>
              <div class="premium-subtitle">七日免费试用!</div>
            </div>
          </div>
  
          <div class="premium-description">
            马上注册并体验所有功能
          </div>
  
          <button class="start-free-button">免费试用</button>
  
          <!-- Navigation Menu -->
          <nav class="sidebar-nav">
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg>
              </div>
              <span>社区</span>
            </a>
            <a href="#" class="nav-item active">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
              </div>
              <span>主页</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
              </div>
              <span>项目</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2H2v10h10V2Z"/><path d="M22 12h-8v10h8V12Z"/><path d="M12 12H2v10h10V12Z"/><path d="M22 2h-8v6h8V2Z"/></svg>
              </div>
              <span>AI工具</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>
              </div>
              <span>资源库</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
              </div>
              <span>API</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6a2 2 0 0 0 2 2h6"/><path d="M22 12v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8"/><circle cx="12" cy="15" r="2"/><path d="M12 13v-2"/></svg>
              </div>
              <span>充值</span>
            </a>
  
            <div class="nav-divider"></div>
  
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
              </div>
              <span>教程</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 19 22 12 13 5 13 19"/><polygon points="2 19 11 12 2 5 2 19"/></svg>
              </div>
              <span>社交软件</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
              </div>
              <span>邀请好友</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>
              </div>
              <span>疑问</span>
            </a>
            <a href="#" class="nav-item">
              <div class="nav-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.36 6.64A9 9 0 0 1 20.77 15"/><path d="M6.16 6.16a9 9 0 1 0 12.68 12.68"/><path d="M12 2v4"/><path d="m2 2 20 20"/></svg>
              </div>
              <span>登出</span>
            </a>
          </nav>
        </div>
      </aside>
  
      <!-- Main Content -->
      <main class="main-content">
        <!-- Feature Highlight with Carousel -->
        <section class="feature-highlight">
          <div class="feature-content">
            <h1 class="feature-title">稳定的图像生成</h1>
            <p class="feature-description">
                只需一些训练图像，您就可以在新的设置、姿势和场景中生成您的角色，同时保持完美、一致的外观。            </p>
            <div class="feature-actions">
              <button class="primary-button">前往工作区</button>
              <button class="secondary-button">如何使用？</button>
            </div>
          </div>
          
          <div class="feature-carousel">
            <div class="carousel-container" ref="carouselContainer">
              <div class="carousel-track" :style="{ transform: `translateX(-${activeSlide * 100}%)` }">
                <div class="carousel-slide" v-for="(slide, index) in carouselSlides" :key="index">
                  <img :src="slide.image" :alt="slide.alt" class="carousel-image">
                </div>
              </div>
            </div>
            <div class="carousel-indicators">
              <button 
                v-for="(slide, index) in carouselSlides" 
                :key="index"
                :class="['indicator', { active: activeSlide === index }]"
                @click="setActiveSlide(index)"
              ></button>
            </div>
          </div>
        </section>
  
        <!-- Recent Projects -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">最近的项目</h2>
            <div class="section-actions">
              <button class="import-button">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Import PSD file
              </button>
              <button class="view-all-button">所有项目</button>
            </div>
          </div>
          
          <div class="projects-grid">
            <div class="project-card start-project">
              <div class="card-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <h3 class="card-title">从一张图片开始</h3>
              <p class="card-description">上传或拖动一张图片到这里</p>
            </div>
            
            <div class="project-card new-project">
              <div class="card-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
              </div>
              <h3 class="card-title">新项目</h3>
              <p class="card-description">创建一个画板</p>
            </div>
            
            <div class="project-card">
              <img src="https://static.dzine.ai/stylar_product/p/16066503/canvas_thumbnail/1745466547022_KBaKj379_.webp" alt="Project thumbnail" class="project-thumbnail">
            </div>
          </div>
        </section>
  
        <!-- Popular Tools -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">热门工具</h2>
            <button class="view-all-button">所有工具</button>
          </div>
          
          <div class="tools-grids">
            <div class="tool-card_a tool-image-to-image">
              <div class="tool-header">
                <div class="tool-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
                </div>
                <span class="tool-name">图生图</span>
              </div>
              <div class="tool-demo">
                <div class="before-image">
                  <span class="tag">前</span>
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-image.png" alt="Before image">
                </div>
                <div class="arrow">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><path d="M15 3h6v6"/><path d="m10 14 11-11"/></svg>
                </div>
                <div class="after-image">
                  <span class="tag">后</span>
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-image.png" alt="After image">
                </div>
              </div>
            </div>
            
            <div class="tool-card_a tool-text-to-image">
              <div class="tool-header">
                <div class="tool-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/></svg>
                </div>
                <span class="tool-name">文生图</span>
              </div>
              <div class="tool-demo">
                <div class="text-prompt">
                  <p>美味的<br>炸芝士<br>Doritos夹心玉米片</p>
                </div>
                <div class="arrow">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><path d="M15 3h6v6"/><path d="m10 14 11-11"/></svg>
                </div>
                <div class="generated-image">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/text-to-image.png" alt="Generated food">
                </div>
              </div>
            </div>
            
            <div class="tool-card_a tool-consistent-character">
              <div class="tool-header">
                <div class="tool-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="5"/><path d="M20 21v-2a5 5 0 0 0-5-5h-6a5 5 0 0 0-5 5v2"/></svg>
                </div>
                <span class="tool-name">固定角色</span>
              </div>
              <div class="tool-demo">
                <div class="character-grid">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/consistent-character.png" alt="Character 1">
                  </div>
              </div>
            </div>
            
            <div class="tool-card_a tool-image-to-video">
              <div class="premium-badge">高级版</div>
              <div class="tool-header">
                <div class="tool-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
                </div>
                <span class="tool-name">图生视频</span>
              </div>
              <div class="tool-demo">
                <div class="video-preview">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-video.png" alt="Video preview">
                  <div class="play-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="white"/></svg>
                  </div>
                </div>
                <div class="video-frames">
                  <img src="https://placeholder.pics/svg/150x40/CCCCCC/666666-CCCCCC/frames" alt="Video frames">
                </div>
              </div>
            </div>
            
            <div class="tool-card_a tool-lip-sync">
              <div class="premium-badge">高级版</div>
              <div class="tool-header">
                <div class="tool-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6v12"/><path d="M6 12h12"/></svg>
                </div>
                <span class="tool-name">口型适配</span>
              </div>
              <div class="tool-demo">
                <div class="video-preview">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-video.png" alt="Lip sync preview">
                  <div class="play-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8" fill="white"/></svg>
                  </div>
                </div>
                <div class="audio-wave">
                  <img src="https://placeholder.pics/svg/150x40/CCCCCC/666666-CCCCCC/audio-wave" alt="Audio wave">
                </div>
              </div>
            </div>
          </div>
        </section>
  
        <!-- Tutorials -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">教程</h2>
          </div>
          
          <div class="tutorials-grid">
            <div class="tutorial-card">
              <div class="tutorial-image">
                <img src="https://static.dzine.ai/tools/dashboard/tutorials/quick-guide-beginner.webp" alt="Quick Guide for Beginners">
              </div>
              <h3 class="tutorial-title">初学者快速教程</h3>
            </div>
            
            <div class="tutorial-card">
              <div class="tutorial-image">
                <img src="https://static.dzine.ai/tools/dashboard/tutorials/HowtoCC.webp" alt="How to Use Consistent Character">
              </div>
              <h3 class="tutorial-title">如何使用固定角色形象</h3>
            </div>
            
            <div class="tutorial-card">
              <div class="tutorial-image">
                <img src="https://static.dzine.ai/tools/dashboard/tutorials/howto-i2i.webp" alt="How to Use Image-to-Image">
              </div>
              <h3 class="tutorial-title">如何使用图生图</h3>
            </div>
            
            <div class="tutorial-card">
              <div class="tutorial-image">
                <img src="https://static.dzine.ai/tools/dashboard/tutorials/How-to-Lip-sync.jpg" alt="How to Use Lip-Sync">
              </div>
              <h3 class="tutorial-title">如何使用口型适配</h3>
            </div>
          </div>
        </section>
  
        <!-- Create Your Own Styles -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">创建我的风格</h2>
            <button class="view-all-button">所有风格</button>
          </div>
          
          <div class="styles-grid">
            <div class="style-card">
              <div class="premium-badge">高级版</div>
              <div class="style-icon">
                <img src="https://static.dzine.ai/assets/panel-swap-bbdd2e37.svg" alt="Quick Style icon">
              </div>
              <h3 class="style-title">快速创建</h3>
            </div>
            
            <div class="style-card">
              <div class="premium-badge">高级版</div>
              <div class="style-icon">
                <img src="https://static.dzine.ai/assets/panel-traning-c1be96fb.svg" alt="Pro Style icon">
              </div>
              <h3 class="style-title">进阶风格</h3>
            </div>
            
            <div class="style-card empty"></div>
            <div class="style-card empty"></div>
          </div>
        </section>
  
        <!-- Game Design -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">游戏美工</h2>
          </div>
          
          <div class="game-design-grid">
            <div class="game-design-card">
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/1da5bf72ad4de3b90b4dea07e4477e9b/1727232343516960_Sketch-to-Image.webp" alt="Sketch to Art">
              </div>
              <h3 class="card-title">草图成图</h3>
            </div>
            
            <div class="game-design-card">
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/02126553a93fc72efeb0d32415b05a9a/1727231459932384_2D-to-3D.webp" alt="2D to 3D Image">
              </div>
              <h3 class="card-title">2D生成3D图像</h3>
            </div>
            
            <div class="game-design-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/a2587147e2b990404086b4c61044ffe8/1727230427074993_Art-Reference.webp" alt="Art Reference">
              </div>
              <h3 class="card-title">画风改变</h3>
            </div>
            
            <div class="game-design-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/5649fcf0f49ae41a64dc6b2d20c3822d/1737013605836734_Prop-Refinement.webp" alt="Refine Details">
              </div>
              <h3 class="card-title">添加细节</h3>
            </div>
            
            <div class="game-design-card">
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/3a6e0e3e62693aeffe2e86d327711fb1/1737013737091523_20250116-154233.jpg.webp" alt="Generate from Text">
              </div>
              <h3 class="card-title">文生图</h3>
            </div>
            
           
          </div>
        </section>
  
        <!-- People Photography -->
        <section class="section">
          <div class="section-header">
            <h2 class="section-title">人像处理</h2>
          </div>
          
          <div class="photography-grid">
            <div class="photography-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/6b251bc0edff6070b3fdc652002c6d81/1727244858324979_Portrait-Enhance.webp" alt="Photography example 1">
              </div>
              <h3 class="card-title">增强图像</h3>
            </div>
            
            <div class="photography-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/fe0de824c87bf5bb508e9c4298f1352d/1727244806410796_Remove-Object.webp" alt="Photography example 2">
              </div>
              <h3 class="card-title">消除物品</h3>
            </div>
            
            <div class="photography-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/4016ad69b170cce555252456bb1dae35/1737013845114467_20250116-154242.jpg.webp" alt="Photography example 3">
                <div class="photo-tag">一件黑色花纹的裙子</div>
              </div>
              <h3 class="card-title">改变花纹</h3>
            </div>
            
            <div class="photography-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/15a6e098c2a4152114a743680a9de1c2/1727244565925145_Add-Object.webp" alt="Photography example 4">
              </div>
              <h3 class="card-title">增加物品</h3>
            </div>
            
            <div class="photography-card">
              <div class="premium-badge">高级版</div>
              <div class="card-image">
                <img src="https://static.dzine.ai/stylar_admin/common/style_avatar/0bbf0b77730d7f0191ecf6f5ba435dc6/1727244476338176_Model-Face-Swap.webp" alt="Photography example 5">
              </div>
              <h3 class="card-title">换脸</h3>
            </div>
            
            
          </div>
        </section>

        
      </main>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  
  // Carousel data
  const carouselSlides = ref([
    { 
      image: 'https://static.dzine.ai/tools/dashboard/heros/Banner-cc.png', 
      alt: 'Character in blue hoodie' 
    },
    { 
      image: 'https://static.dzine.ai/tools/dashboard/heros/Banner-cc.png', 
      alt: 'Character playing tennis' 
    },
    { 
      image: 'https://static.dzine.ai/tools/dashboard/heros/lip-sync-Banner.png', 
      alt: 'Character swimming' 
    }
  ]);
  
  const activeSlide = ref(0);
  
  const setActiveSlide = (index) => {
    activeSlide.value = index;
  };
  
  // Auto-rotate carousel
  onMounted(() => {
    const interval = setInterval(() => {
      activeSlide.value = (activeSlide.value + 1) % carouselSlides.value.length;
    }, 5000);
  
    return () => clearInterval(interval);
  });
  </script>
  
  <style scoped>
  /* Layout */
  .dashboard-layout {
    display: flex;
    min-height: 100vh;
    background-color: #0e0e0e;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  }
  
  /* Sidebar */
  .sidebar {
    width: 350px;
    background-color: #121212;
    border-right: 1px solid #222;
    display: flex;
    flex-direction: column;
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    overflow-y: auto;
    z-index: 10;
  }
  
  .sidebar-header {
    padding: 1.5rem;
    border-bottom: 1px solid #222;
  }
  
  .logo {
    display: flex;
    align-items: center;
    text-decoration: none;
  }
  
  .logo-img {
    height: 60px;
    
    
  }
  
  .beta-tag {
    font-size: 0.6rem;
    color: #f5c518;
    margin-left: 4px;
    font-weight: 600;
  }
  
  .sidebar-content {
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .user-profile {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #222;
  }
  
  .avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    background-color: #333;
  }
  
  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .user-info {
    flex: 1;
  }
  
  .user-id {
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  
  .chevron {
    color: #666;
  }
  
  .user-plan {
    color: #888;
    font-size: 0.9rem;
  }
  
  .credits-section {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background-color: #1a1a1a;
    border-radius: 8px;
  }
  
  .credits-icon {
    color: #f5c518;
  }
  
  .credits-label {
    flex: 1;
    color: #ccc;
  }
  
  .credits-value {
    font-weight: 600;
    font-size: 1.1rem;
  }
  
  .premium-promo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  .crown-icon {
    color: #f5c518;
  }
  
  .premium-title {
    font-weight: 600;
  }
  
  .premium-subtitle {
    color: #f5c518;
    font-weight: 700;
  }
  
  .premium-description {
    color: #888;
    font-size: 0.9rem;
    line-height: 1.4;
    margin-bottom: 1rem;
  }
  
  .start-free-button {
    width: 100%;
    padding: 0.75rem;
    background-color: #f5c518;
    color: #000;
    border: none;
    border-radius: 2rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-bottom: 1.5rem;
  }
  
  .start-free-button:hover {
    background-color: #e6b400;
  }
  
  .sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    border-radius: 8px;
    color: #ccc;
    text-decoration: none;
    transition: all 0.2s ease;
  }
  
  .nav-item:hover {
    background-color: #1a1a1a;
    color: #fff;
  }
  
  .nav-item.active {
    background-color: #1a1a1a;
    color: #fff;
  }
  
  .nav-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
  }
  
  .nav-divider {
    height: 1px;
    background-color: #222;
    margin: 0.5rem 0;
  }
  
  /* Main Content */
  .main-content {
    flex: 1;
    margin-left: 350px;
    padding: 2rem;
    overflow-y: auto;
  }
  
  /* Feature Highlight */
  .feature-highlight {
    display: flex;
    background-color: #1a1a1a;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2rem;
  }
  
  .feature-content {
    flex: 1;
    padding: 2.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  
  .feature-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: left;
  }
  
  .feature-description {
    color: #ccc;
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 500px;
    text-align: left;
  }
  
  .feature-actions {
    display: flex;
    gap: 1rem;
  }
  
  .primary-button {
    padding: 0.75rem 1.5rem;
    background-color: #f5c518;
    color: #000;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .primary-button:hover {
    background-color: #e6b400;
  }
  
  .secondary-button {
    padding: 0.75rem 1.5rem;
    background-color: #333;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .secondary-button:hover {
    background-color: #444;
  }
  
  .feature-carousel {
    flex: 1;
    position: relative;
  }
  
  .carousel-container {
    width: 100%;
    height: 100%;
    overflow: hidden;
  }
  
  .carousel-track {
    display: flex;
    height: 100%;
    transition: transform 0.5s ease;
  }
  
  .carousel-slide {
    min-width: 100%;
    height: 100%;
  }
  
  .carousel-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .carousel-indicators {
    position: absolute;
    bottom: 20px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 0.5rem;
  }
  
  .indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: rgba(255, 255, 255, 0.3);
    border: none;
    padding: 0;
    cursor: pointer;
  }
  
  .indicator.active {
    background-color: #f5c518;
    transform: scale(1.2);
  }
  
  /* Section Styles */
  .section {
    margin-bottom: 3rem;
  }
  
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }
  
  .section-title {
    font-size: 1.75rem;
    font-weight: 700;
  }
  
  .section-actions {
    display: flex;
    gap: 1rem;
  }
  
  .import-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background-color: #333;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .import-button:hover {
    background-color: #444;
  }
  
  .view-all-button {
    padding: 0.5rem 1rem;
    background-color: transparent;
    color: #fff;
    border: 1px solid #333;
    border-radius: 8px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .view-all-button:hover {
    background-color: #222;
  }
  
  /* Projects Grid */
  .projects-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
  }
  
  .project-card {
    height: 280px;
    background-color: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    cursor: pointer;
  }
  
  .project-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  
  .card-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-color: rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    color: #f5c518;
  }
  
  .card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  
  .card-description {
    color: #888;
    font-size: 0.9rem;
  }
  
  .project-thumbnail {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  /* Tools Grid */
  .tools-grids {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1.5rem;
  }
  
  .tool-card_a {
    height: 280px;
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.2s ease;
    cursor: pointer;
  }
  
  .tool-card_a:hover {
    transform: translateY(-5px);
  }
  
  .tool-image-to-image {
    background: linear-gradient(135deg, #291e00 0%, #3a2900 100%);
  }
  
  .tool-text-to-image {
    background: linear-gradient(135deg, #3a1700 0%, #4d1d00 100%);
  }
  
  .tool-consistent-character {
    background: linear-gradient(135deg, #002a1c 0%, #00402b 100%);
  }
  
  .tool-image-to-video {
    background: linear-gradient(135deg, #001a40 0%, #002966 100%);
  }
  
  .tool-lip-sync {
    background: linear-gradient(135deg, #29004d 0%, #3d0073 100%);
  }
  
  .premium-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background-color: #f5c518;
    color: #000;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  
  .tool-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }
  
  .tool-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .tool-name {
    font-weight: 600;
  }
  
  .tool-demo {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  
  .before-image, .after-image, .text-prompt, .generated-image {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
  }
  
  .tag {
    position: absolute;
    top: 5px;
    left: 5px;
    background-color: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    z-index: 1;
  }
  
  .arrow {
    margin: 0 0.5rem;
    color: rgba(255, 255, 255, 0.8);
  }
  
  .text-prompt {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.75rem;
    border-radius: 8px;
    text-align: left;
  }
  
  .text-prompt p {
    margin: 0;
    color: white;
    font-size: 0.8rem;
    line-height: 1.4;
  }
  
  .character-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
  }
  
  .video-preview {
    position: relative;
    margin-bottom: 0.75rem;
  }
  
  .play-button {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    cursor: pointer;
  }
  
  .video-frames, .audio-wave {
    border-radius: 8px;
    overflow: hidden;
  }
  
  /* Tutorials Grid */
  .tutorials-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
  }
  
  .tutorial-card {
    background-color: #fff;
    border-radius: 20px;
    overflow: hidden;
    transition: all 0.2s ease;
    cursor: pointer;
    padding: 1rem;
  }
  
  .tutorial-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  
  .tutorial-image {
    height: 180px;
    overflow: hidden;
    border-radius: 1rem;
  }
  
  .tutorial-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .tutorial-title {
    padding: 1rem;
    color: #000;
    font-size: 2.3rem;
    font-weight: 600;
    text-align: left;
    line-height: 2.6rem;
  }
  
  /* Styles Grid */
  .styles-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
  }
  
  .style-card {
    height: 180px;
    background-color: #1a1a1a;
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    transition: all 0.2s ease;
    cursor: pointer;
  }
  
  .style-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  
  .style-card.empty {
    border: 2px dashed #333;
    background-color: transparent;
  }
  
  .style-icon {
    margin-bottom: 1rem;
  }
  
  .style-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  
  .style-description {
    color: #888;
    font-size: 0.9rem;
    line-height: 1.4;
  }
  
  /* Game Design Grid */
  .game-design-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
  }
  
  .game-design-card {
    background-color: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
  }
  
  .game-design-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  
  .card-image {
    height: 180px;
    overflow: hidden;
  }
  
  .card-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .card-title {
    padding: 1rem;
    font-size: 1.1rem;
    font-weight: 600;
  }
  
  /* Photography Grid */
  .photography-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
  }
  
  .photography-card {
    background-color: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
  }
  
  .photography-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  
  .photo-tag {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }
  
  /* Responsive Styles */
  @media (max-width: 1200px) {
    .feature-highlight {
      flex-direction: column;
    }
    
    .feature-content {
      padding: 2rem;
    }
  }
  
  @media (max-width: 992px) {
    .sidebar {
      width: 220px;
    }
    
    .main-content {
      margin-left: 220px;
    }
    
    .tools-grids, .game-design-grid, .photography-grid {
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }
  }
  
  @media (max-width: 768px) {
    .sidebar {
      width: 0;
      transform: translateX(-100%);
      transition: transform 0.3s ease;
    }
    
    .sidebar.open {
      width: 260px;
      transform: translateX(0);
    }
    
    .main-content {
      margin-left: 0;
    }
    
    .feature-title {
      font-size: 2rem;
    }
    
    .feature-description {
      font-size: 1rem;
    }
  }
  
  @media (max-width: 576px) {
    .main-content {
      padding: 1rem;
    }
    
    .section-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }
    
    .feature-actions {
      flex-direction: column;
      width: 100%;
    }
    
    .primary-button, .secondary-button {
      width: 100%;
    }
  }
  </style>