<template>
  <div class="app">
    <!-- Navigation -->
    <nav class="navbar">
      <div class="navbar-left">
        <a href="#" class="logo">
          siyan<span class="beta">BETA</span>
        </a>

        <div class="nav-links">
          <div class="dropdown">
            <button class="dropdown-btn">
              工具 <span class="dropdown-arrow">▼</span>
            </button>
          </div>

          <div class="dropdown">
            <button class="dropdown-btn">
              资源 <span class="dropdown-arrow">▼</span>
            </button>
          </div>

          <a href="#pricing" class="nav-link">定价</a>
          <a href="#api" class="nav-link">API</a>
          <a href="#contact" class="nav-link">商务联系</a>
          <a href="#demo" class="nav-link">体验版</a>
        </div>
      </div>

      <div class="navbar-right">
        <a href="#" class="nav-link" @click="showLoginModal = true">登录</a>
        <button class="btn-primary">免费试用</button>
      </div>

      <button class="mobile-menu-btn" @click="mobileMenuOpen = !mobileMenuOpen">
        <span></span>
        <span></span>
        <span></span>
      </button>
    </nav>

    <!-- Mobile Menu -->
    <div class="mobile-menu" :class="{ 'open': mobileMenuOpen }">
      <a href="#" class="mobile-link">工具</a>
      <a href="#" class="mobile-link">资源</a>
      <a href="#pricing" class="mobile-link">价格</a>
      <a href="#api" class="mobile-link">API</a>
      <a href="#contact" class="mobile-link">商务联系</a>
      <a href="#demo" class="mobile-link">体验版</a>
      <a href="#" class="mobile-link" @click="showLoginModal = true">登录</a>
      <button class="btn-primary mobile-btn">免费试用</button>
    </div>

    <!-- Hero Section -->
    <section class="hero">
      <div class="container">
        <h1 class="hero-title" >
          一次性获取所有<br>
          AI图像工具
        </h1>
        <p class="hero-subtitle">
          节省您的时间和金钱：不再需要在昂贵的AI设计应用程序之间切换        
        </p>
        <button class="hero-button">
          今日 <strong>免费</strong> 试用
        </button>
      </div>
    </section>

    <!-- Tools Grid -->
    <section class="tools-grid">
      <div class="container">
        <div class="tools-container">
          <!-- Row 1 -->
          <div class="tool-card tool-image-to-image">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect width="18" height="18" x="3" y="3" rx="2" />
                  <circle cx="9" cy="9" r="2" />
                  <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                </svg>
              </div>
              <h3 class="tool-name">图生图</h3>
            </div>
            <div class="tool-content">
              <div class="demo-container">
                <div class="before-image">
                  <span class="tag">原图</span>
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-image.png" alt="Sketch">
                </div>
                <div class="transform-arrow">
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    <path d="M15 3h6v6" />
                    <path d="m10 14 11-11" />
                  </svg>
                </div>
                <div class="after-image">
                  <span class="tag">生图</span>
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-image.png" alt="Generated Photo">
                </div>
              </div>
            </div>
          </div>

          <div class="tool-card tool-text-to-image">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M4 7V4h16v3" />
                  <path d="M9 20h6" />
                  <path d="M12 4v16" />
                </svg>
              </div>
              <h3 class="tool-name">文转图</h3>
            </div>
            <div class="tool-content">
              <div class="demo-container">
                <div class="text-prompt">
                  <p>西湖<br>醋鱼</p>
                </div>
                <div class="transform-arrow">
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    <path d="M15 3h6v6" />
                    <path d="m10 14 11-11" />
                  </svg>
                </div>
                <div class="generated-image">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/text-to-image.png" alt="Generated Food">
                </div>
              </div>
            </div>
          </div>

          <div class="tool-card tool-consistent-character">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="8" r="5" />
                  <path d="M20 21v-2a5 5 0 0 0-5-5h-6a5 5 0 0 0-5 5v2" />
                </svg>
              </div>
              <h3 class="tool-name">固定角色</h3>
            </div>
            <div class="tool-content">
              <div class="character-grid">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/consistent-character.png" alt="Character 1">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/consistent-character.png" alt="Character 2">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/consistent-character.png" alt="Character 3">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/consistent-character.png" alt="Character 4">
              </div>
            </div>
          </div>

          <div class="tool-card tool-image-to-video">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="m22 8-6 4 6 4V8Z" />
                  <rect width="14" height="12" x="2" y="6" rx="2" ry="2" />
                </svg>
              </div>
              <h3 class="tool-name">图生视频</h3>
            </div>
            <div class="tool-content">
              <div class="video-demo">
                <div class="video-preview">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/image-to-video.png" alt="Video Preview">
                  <div class="play-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="white"
                      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <polygon points="10 8 16 12 10 16 10 8" fill="white" />
                    </svg>
                  </div>
                </div>
                <div class="video-frames">
                  <img src="https://placeholder.pics/svg/200x60/CCCCCC/666666-CCCCCC/frames" alt="Video Frames">
                </div>
              </div>
            </div>
          </div>

          <!-- Row 2 -->
          <div class="tool-card tool-insert-object">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10" stroke-dasharray="4 4" />
                  <line x1="12" y1="8" x2="12" y2="16" />
                  <line x1="8" y1="12" x2="16" y2="12" />
                </svg>
              </div>
              <h3 class="tool-name">插入物品</h3>
            </div>
            <div class="tool-content">
              <div class="demo-container">
                <div class="before-image">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/insert-object.png" alt="Dog">
                </div>
                <div class="transform-arrow">
                  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    <path d="M15 3h6v6" />
                    <path d="m10 14 11-11" />
                  </svg>
                </div>
                
                <div class="after-image">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/insert-object.png" alt="Dog with Hat">
                </div>
              </div>
            </div>
          </div>

          <div class="tool-card tool-ai-eraser">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="m20 20-6-6m2-4a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z" />
                </svg>
              </div>
              <h3 class="tool-name">AI擦除</h3>
            </div>
            <div class="tool-content">
              <div class="eraser-demo">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/ai-eraser.png" alt="Dog with Eraser">
                <div class="eraser-circle">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                    stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 3h18v18H3z" />
                    <path d="M15 9h.01M9 15h.01" />
                    <path d="m15 15-6-6" />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          <div class="tool-card tool-expand">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M15 3h6v6" />
                  <path d="M9 21H3v-6" />
                  <path d="m21 3-7 7" />
                  <path d="m3 21 7-7" />
                </svg>
              </div>
              <h3 class="tool-name">扩展</h3>
            </div>
            <div class="tool-content">
              <div class="expand-demo">
                <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/expand.png" alt="Portrait">
                <div class="expand-arrows">
                  <div class="arrow left">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                      stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="m15 18-6-6 6-6" />
                    </svg>
                  </div>
                  <div class="arrow right">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                      stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="m9 18 6-6-6-6" />
                    </svg>
                  </div>
                  <div class="corner top-left"></div>
                  <div class="corner top-right"></div>
                  <div class="corner bottom-left"></div>
                  <div class="corner bottom-right"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="tool-card tool-enhance">
            <div class="tool-header">
              <div class="tool-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="m2 12 5.25 5 2-2-5.3-5.3a1 1 0 0 0-1.4 0l-.55.6a1 1 0 0 0 0 1.7Z" />
                  <path d="M15 12 9 6l-2 2 6 6 2-2Z" />
                  <path
                    d="m22 12-5.3-5.3a1 1 0 0 0-1.4 0l-5.3 5.3 2 2 5.25-5 4.25 4a1 1 0 0 0 1.4-.3l.6-.5a1 1 0 0 0-.5-1.2Z" />
                </svg>
              </div>
              <h3 class="tool-name">增强</h3>
            </div>
            <div class="tool-content">
              <div class="enhance-demo">
                <div class="split-image">
                  <img src="https://static.dzine.ai/tools/dashboard/ai-tools-cover/enhance.png"
                    alt="Pixelated vs Enhanced">
                  <div class="split-line"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

       
      </div>
    </section>

    <section class="hero">
      <h1 class="hero-title">
        一体化<br />
        AI设计平台
      </h1>
      <p class="hero-subtitle">
        在一个平台即可完成任意人工智能生成和编辑。      
      </p>
    </section>
    <div class="feature-sections">
      <!-- Section 1: AI Photo Filter -->
      <section class="feature-section">
        <div class="feature-container">
          <div class="feature-image">
                        <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/AI%2BPhoto%2BFilter1.mp4" type="video/mp4" data-v-5f31b5ed=""></video>

          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI照片风格</h2>
            <p class="feature-description">
              使用我们的 <strong>AI滤镜</strong> ，将您的肖像或照片转换为卡通。我们的工具可以轻松地在生成艺术中保留你的照片细节和人物。
            </p>
            <a href="#" class="feature-button">改变你的照片风格</a>
          </div>
        </div>
      </section>

      <!-- Section 2: Combine Images with AI -->
      <section class="feature-section feature-section-reverse">
        <div class="feature-container">
          <div class="feature-image feature-image-colored">
            <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/Combine+Images+with+AI.mp4" type="video/mp4" data-v-5f31b5ed=""></video>
          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI图像组合</h2>
            <p class="feature-description">
              使用 <strong>AI图像组合</strong> 生成式地将不同的元素合并为一个完整的图像，并使它们在风格上都匹配。.
            </p>
            <a href="#" class="feature-button">组合你的图片</a>
          </div>

        </div>
      </section>

      <!-- Section 3: AI Background Removal -->
      <section class="feature-section">
        <div class="feature-container">
          <div class="feature-image">
            <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/Remove+Background.mp4" type="video/mp4" data-v-5f31b5ed=""></video>
          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI抠图</h2>
            <p class="feature-description">
              使用我们的 <strong>AI 抠图工具</strong> 完美地从您的图像中移除背景。
            </p>
            <a href="#" class="feature-button">快速移除背景</a>
          </div>
        </div>
      </section>

      <!-- Section 4: AI Image Enhancer -->
      <section class="feature-section feature-section-reverse">
        <div class="feature-container">
          <div class="feature-image feature-image-colored" style="--feature-bg-color: #D1C4E9;">
            <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/Remove+Background.mp4" type="video/mp4" data-v-5f31b5ed=""></video>
          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI图像增强器</h2>
            <p class="feature-description">
              使用我们的 <strong>AI图像增强器</strong> 将低质量图像转换为晶莹剔透的高分辨率视觉效果。
            </p>
            <a href="#" class="feature-button">增强图像</a>
          </div>

        </div>
      </section>

      <!-- Section 5: AI Text to Image -->
      <section class="feature-section">
        <div class="feature-container">
          <div class="feature-image">
            <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/text-effect-logo-1.mp4" type="video/mp4" data-v-5f31b5ed=""></video>
          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI文生图</h2>
            <p class="feature-description">
              我们的 <strong>AI文生图工具</strong>
              只需一个文本提示，将您的想法转化为令人惊叹的视觉效果。            
            </p>
            <a href="#" class="feature-button">Generate from text</a>
          </div>
        </div>
      </section>

      <!-- Section 6: AI Image Expansion -->
      <section class="feature-section feature-section-reverse">
        <div class="feature-container">
          <div class="feature-image feature-image-colored" style="--feature-bg-color: #BBDEFB;">
            <video class="vi" poster="https://static.dzine.ai/assets/pre_load-dz-800-400.jpeg" playsinline="" autoplay="" loop="" muted="" data-v-5f31b5ed=""><source src="https://static.dzine.ai/videos/generative-object-1.mp4" type="video/mp4" data-v-5f31b5ed=""></video>
          </div>
          <div class="feature-content">
            <h2 class="feature-title">AI图片扩展</h2>
            <p class="feature-description">
              使用我们的 <strong>AI图片扩展工具</strong> 将您的图像扩展到原始边界之外。</p>
            <a href="#" class="feature-button">扩展画布</a>
          </div>

        </div>
      </section>
    </div>
    <!-- Features Section -->
    <section class="features">
      <h2 class="section-title">为什么选择我们？</h2>

      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              class="lucide lucide-zap">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
          </div>
          <h3 class="feature-title">快速生成</h3>
          <p class="feature-description">在几秒内生成高质量的视觉效果</p>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              class="lucide lucide-sparkles">
              <path
                d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z">
              </path>
              <path d="M5 3v4"></path>
              <path d="M19 17v4"></path>
              <path d="M3 5h4"></path>
              <path d="M17 19h4"></path>
            </svg>
          </div>
          <h3 class="feature-title">卓越品质</h3>
          <p class="feature-description">不需要专业技能就能创造出专业级别的视觉效果</p>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              class="lucide lucide-wallet">
              <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"></path>
              <path d="M3 5v14a2 2 0 0 0 2 2h16v-5"></path>
              <path d="M18 12a2 2 0 0 0 0 4h4v-4Z"></path>
            </svg>
          </div>
          <h3 class="feature-title">高性价比</h3>
          <p class="feature-description">一键订购所有工具，为您节省开销</p>
        </div>
      </div>
    </section>

    <!-- Pricing Section -->
    <section id="pricing" class="pricing">
      <h2 class="section-title">定价透明公开</h2>
      <p class="section-subtitle">根据需求选择最适合的套餐</p>

      <div class="pricing-grid">
        <div class="pricing-card">
          <div class="pricing-header">
            <h3 class="pricing-title">Basic</h3>
            <div class="pricing-price">
              <span class="price">$19</span>
              <span class="period">/month</span>
            </div>
          </div>
          <ul class="pricing-features">
            <li>5 AI generations per day</li>
            <li>Access to basic tools</li>
            <li>720p resolution</li>
            <li>Email support</li>
          </ul>
          <button class="btn-secondary">Get Started</button>
        </div>

        <div class="pricing-card featured">
          <div class="pricing-badge">Popular</div>
          <div class="pricing-header">
            <h3 class="pricing-title">Pro</h3>
            <div class="pricing-price">
              <span class="price">$49</span>
              <span class="period">/month</span>
            </div>
          </div>
          <ul class="pricing-features">
            <li>50 AI generations per day</li>
            <li>Access to all tools</li>
            <li>4K resolution</li>
            <li>Priority support</li>
            <li>Commercial usage</li>
          </ul>
          <button class="btn-primary">Get Started</button>
        </div>

        <div class="pricing-card">
          <div class="pricing-header">
            <h3 class="pricing-title">Enterprise</h3>
            <div class="pricing-price">
              <span class="price">Custom</span>
            </div>
          </div>
          <ul class="pricing-features">
            <li>Unlimited AI generations</li>
            <li>Access to all tools + API</li>
            <li>8K resolution</li>
            <li>Dedicated support</li>
            <li>Custom integrations</li>
            <li>Team management</li>
          </ul>
          <button class="btn-secondary">Contact Sales</button>
        </div>
      </div>
    </section>
    <section class="use-cases-section">
      <div class="container">
        <h2 class="use-cases-title">AI 图像生成使用秘籍</h2>
        <p class="use-cases-subtitle">专业人士怎么使用我们的工具？</p>

        <div class="use-cases-tabs">
          <button v-for="(tab, index) in tabs" :key="index" :class="['tab-button', { active: activeTab === index }]"
            @click="setActiveTab(index)">
            {{ tab.name }}
          </button>
        </div>

        <div class="use-cases-content">
          <div class="content-left">
            <h3 class="content-title">{{ tabs[activeTab].title }}</h3>
            <p class="content-description">{{ tabs[activeTab].description }}</p>
          </div>

          <div class="content-right">
            <div class="image-carousel">
              <div class="carousel-images" :style="{ transform: `translateX(-${activeImageIndex * 100}%)` }">
                <img v-for="(image, index) in tabs[activeTab].images" :key="index" :src="image"
                  :alt="`${tabs[activeTab].name} example ${index + 1}`" class="carousel-image" />
              </div>

              <div class="carousel-controls" v-if="tabs[activeTab].images.length > 1">
                <button class="control-button prev" @click="prevImage" :disabled="activeImageIndex === 0">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 18l-6-6 6-6" />
                  </svg>
                </button>
                <div class="carousel-dots">
                  <button v-for="(_, index) in tabs[activeTab].images" :key="index"
                    :class="['dot', { active: activeImageIndex === index }]" @click="setActiveImage(index)"></button>
                </div>
                <button class="control-button next" @click="nextImage"
                  :disabled="activeImageIndex === tabs[activeTab].images.length - 1">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <!-- Testimonials -->
    <section class="testimonials">
      <h2 class="section-title">用户评价</h2>

      <div class="testimonials-grid">
        <div class="testimonial-card">
          <div class="testimonial-content">
            "工具完全改变了我们的创意工作流程。过去需要数小时才能完成的内容，现在只需要几分钟。"
          </div>
          <div class="testimonial-author">
            <div class="testimonial-avatar"></div>
            <div class="testimonial-info">
              <div class="testimonial-name">Sarah Johnson</div>
              <div class="testimonial-role">Marketing Director</div>
            </div>
          </div>
        </div>

        <div class="testimonial-card">
          <div class="testimonial-content">
            "使用这个简单的工具，我就像有一个全天候待命的专业设计师。"
          </div>
          <div class="testimonial-author">
            <div class="testimonial-avatar"></div>
            <div class="testimonial-info">
              <div class="testimonial-name">Michael Chen</div>
              <div class="testimonial-role">E-commerce Entrepreneur</div>
            </div>
          </div>
        </div>

        <div class="testimonial-card">
          <div class="testimonial-content">
            "使用这个工具，我们将设计成本降低了70%，同时提高了30% 的生产量。"
          </div>
          <div class="testimonial-author">
            <div class="testimonial-avatar"></div>
            <div class="testimonial-info">
              <div class="testimonial-name">Alex Rivera</div>
              <div class="testimonial-role">Creative Director</div>
            </div>
          </div>
        </div>
      </div>
    </section>



    <!-- CTA Section -->
    <section class="cta">
      <h2 class="cta-title">准备好进行一场视觉体验了吗？</h2>
      <p class="cta-subtitle">加入成千上万的创作者和企业</p>
         <!-- <button class="btn-primary btn-large">免费试用</button>
  <p class="cta-note">No credit card required</p>-->
    </section>

    <!-- New Footer Section Based on Image -->
    <div class="pre-footer">
      <div class="container">
        <h2 class="pre-footer-title">着手创建你的第一个项目！</h2>
        <button class="btn-primary btn-large">免费试用</button>
      </div>
    </div>

    <!-- New Footer with Columns -->
    <footer class="new-footer">
      <div class="container">
        <div class="footer-top">
          <div class="footer-brand">
            <a href="#" class="footer-logo">
              <span class="logo-text">siyan</span>
              <span class="logo-beta">BETA</span>
            </a>
            <p class="footer-tagline">
              Controllable AI Image Editing &<br>
              Graphic Design Tool for Everyone
            </p>
            <div class="footer-social">
              <a href="#" class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path
                    d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z">
                  </path>
                </svg>
              </a>
              <a href="#" class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect width="20" height="20" x="2" y="2" rx="5" ry="5"></rect>
                  <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                  <line x1="17.5" x2="17.51" y1="6.5" y2="6.5"></line>
                </svg>
              </a>
              <a href="#" class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path
                    d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z">
                  </path>
                  <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon>
                </svg>
              </a>
              <a href="#" class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                  <rect width="4" height="12" x="2" y="9"></rect>
                  <circle cx="4" cy="4" r="2"></circle>
                </svg>
              </a>
              <a href="#" class="social-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <path d="M14.5 9a3.5 3.5 0 0 0-5 0"></path>
                  <path d="M9 12a3 3 0 0 0 6 0"></path>
                  <path d="M14.5 15a3.5 3.5 0 0 0-5 0"></path>
                </svg>
              </a>
            </div>
          </div>

          <div class="footer-links-section">
            <div class="footer-links-column">
              <h3 class="footer-column-title">AI PHOTO FILTER</h3>
              <ul class="footer-links">
                <li><a href="#">2D to 3D Converter</a></li>
                <li><a href="#">AI Design Sketch</a></li>
                <li><a href="#">Image to Prompt</a></li>
                <li><a href="#">AI Style Transfer</a></li>
                <li><a href="#">Turn Sketch to Art</a></li>
                <li><a href="#">AI Anime Filter</a></li>
                <li><a href="#">All Tools</a></li>
              </ul>
            </div>

            <div class="footer-links-column">
              <h3 class="footer-column-title">AI IMAGE EDITOR</h3>
              <ul class="footer-links">
                <li><a href="#">Add Object into Image</a></li>
                <li><a href="#">Remove Object from Image</a></li>
                <li><a href="#">Remove Background</a></li>
                <li><a href="#">AI Combine Images</a></li>
                <li><a href="#">AI Photo Enhancer</a></li>
                <li><a href="#">AI Photo Expand</a></li>
                <li><a href="#">Vectorize Image</a></li>
                <li><a href="#">Insert Object</a></li>
              </ul>
            </div>

            <div class="footer-links-column">
              <h3 class="footer-column-title">AI ART GENERATOR</h3>
              <ul class="footer-links">
                <li><a href="#">AI Character Generator</a></li>
                <li><a href="#">AI Anime Generator</a></li>
                <li><a href="#">AI Comic Generator</a></li>
                <li><a href="#">Flux.1 AI Image Generator</a></li>
                <li><a href="#">AI Coloring Book Generator</a></li>
                <li><a href="#">AI Vector Image Generator</a></li>
                <li><a href="#">AI Product Background Generator</a></li>
              </ul>
            </div>

            <div class="footer-links-column">
              <h3 class="footer-column-title">RESOURCES</h3>
              <ul class="footer-links">
                <li><a href="#">Blog</a></li>
                <li><a href="#">Tutorial</a></li>
                <li><a href="#">Affiliate Program</a></li>
                <li><a href="#">Educator Program</a></li>
                <li><a href="#">FAQ</a></li>
              </ul>
            </div>

            <div class="footer-links-column">
              <h3 class="footer-column-title">PRICING</h3>
              <ul class="footer-links">
                <li><a href="#">Plan</a></li>
                <li><a href="#">Enterprise</a></li>
                <li><a href="#">Contact Sales</a></li>
              </ul>
            </div>

            <div class="footer-links-column">
              <h3 class="footer-column-title">ABOUT</h3>
              <ul class="footer-links">
                <li><a href="#">Contact</a></li>
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Term of Use</a></li>
              </ul>
            </div>
          </div>
        </div>

        <div class="footer-bottom">
          <p class="copyright">Copyright © {{ new Date().getFullYear() }} siyan</p>
        </div>
      </div>
    </footer>

    <!-- Login Modal -->
    <div class="modal-overlay" v-if="showLoginModal" @click="closeLoginModal"></div>
    <div class="login-modal" v-if="showLoginModal">
      <div class="modal-header">
        <h3 class="modal-title">Log in to siyan.ai</h3>
        <button class="modal-close" @click="showLoginModal = false">×</button>
      </div>

      <div class="modal-body">
        <form @submit.prevent="handleLogin" class="login-form">
          <div class="form-group">
            <label for="email">Email</label>
            <input type="email" id="email" v-model="loginForm.email" placeholder="your@email.com" required>
          </div>

          <div class="form-group">
            <label for="password">Password</label>
            <div class="password-input">
              <input :type="showPassword ? 'text' : 'password'" id="password" v-model="loginForm.password"
                placeholder="Enter your password" required>
              <button type="button" class="password-toggle" @click="showPassword = !showPassword">
                <svg v-if="!showPassword" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                  class="lucide lucide-eye">
                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                  class="lucide lucide-eye-off">
                  <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"></path>
                  <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"></path>
                  <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"></path>
                  <line x1="2" x2="22" y1="2" y2="22"></line>
                </svg>
              </button>
            </div>
          </div>

          <div class="form-options">
            <label class="checkbox-container">
              <input type="checkbox" v-model="loginForm.rememberMe">
              <span class="checkmark"></span>
              Remember me
            </label>
            <a href="#" class="forgot-password">Forgot password?</a>
          </div>

          <button type="submit" class="btn-primary btn-full">Log in</button>
        </form>

        <div class="divider">
          <span>OR</span>
        </div>

        <div class="social-login">
          <button class="btn-social google">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              class="lucide lucide-mail">
              <rect width="20" height="16" x="2" y="4" rx="2"></rect>
              <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"></path>
            </svg>
            Continue with Google
          </button>

          <button class="btn-social github">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
              class="lucide lucide-github">
              <path
                d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4">
              </path>
              <path d="M9 18c-4.51 2-5-2-7-2"></path>
            </svg>
            Continue with GitHub
          </button>
        </div>

        <p class="signup-prompt">
          Don't have an account? <a href="#">Sign up</a>
        </p>
      </div>
    </div>

    <!-- Promo Popup -->
    <div class="promo-popup" v-if="showPromoPopup">
      <button class="promo-close" @click="showPromoPopup = false">×</button>

      <div class="promo-content">
        <div class="promo-decoration top-right"></div>
        <div class="promo-decoration bottom-left"></div>

        <h3 class="promo-title">Spring Surprise</h3>
        <div class="promo-discount">
          <span class="discount-number">25</span>
          <span class="discount-percent">%</span>
        </div>
        <div class="discount-label">OFF</div>
        <p class="promo-subtitle">Your Next Payment</p>

        <div class="promo-code">
          <span class="code-label">Use code</span>
          <span class="code">APRIL25</span>
          <span class="code-label">at checkout</span>
        </div>

        <p class="promo-note">Hurry — offer valid for a limited time only!</p>

        <div class="promo-actions">
          <button class="btn-outline" @click="showPromoPopup = false">Not now</button>
          <button class="btn-primary">Learn more</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';


// State
const mobileMenuOpen = ref(false);
const showLoginModal = ref(false);
const showPassword = ref(false);
const showPromoPopup = ref(false);
const activeTab = ref(0);
const activeImageIndex = ref(0);

const loginForm = ref({
  email: '',
  password: '',
  rememberMe: false
});

// Tools data
const tools = ref([
  {
    id: 1,
    title: "Image-to-Image",
    icon: 'svg-image',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-green",
    isHovered: false
  },
  {
    id: 2,
    title: "Text-to-Image",
    icon: 'svg-text',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-amber",
    isHovered: false
  },
  {
    id: 3,
    title: "Face Swap",
    icon: 'svg-user',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-purple",
    isHovered: false
  },
  {
    id: 4,
    title: "Local Edit",
    icon: 'svg-edit',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-green",
    isHovered: false
  },
  {
    id: 5,
    title: "Insert Object",
    icon: 'svg-plus',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-green",
    isHovered: false
  },
  {
    id: 6,
    title: "AI Eraser",
    icon: 'svg-eraser',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-blue",
    isHovered: false
  },
  {
    id: 7,
    title: "Expand",
    icon: 'svg-expand',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-purple",
    isHovered: false
  },
  {
    id: 8,
    title: "Enhance",
    icon: 'svg-wand',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-green",
    isHovered: false
  },
  {
    id: 9,
    title: "Product Background",
    icon: 'svg-bag',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-amber",
    isHovered: false
  },
  {
    id: 10,
    title: "Image-to-3D",
    icon: 'svg-cube',
    beforeImage: "https://placeholder.pics/svg/112x112",
    afterImage: "https://placeholder.pics/svg/112x112/DEDEDE-DEDEDE",
    bgColor: "bg-green",
    isHovered: false
  }
]);

// Methods
const closeLoginModal = (e) => {
  if (e.target.classList.contains('modal-overlay')) {
    showLoginModal.value = false;
  }
};

const handleLogin = () => {
  // Here you would typically handle the login logic
  console.log('Login submitted:', loginForm.value);
  alert('Login successful!');
  showLoginModal.value = false;
};

const tabs = ref([
  {
    name: "人物设计",
    title: "生成动画角色",
    description: "借助AI辅助工具，设计师可以快速为动画和游戏创建和定制角色，包括将2D草图转换为动态3D角色。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/character-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/character-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/character-design.png"
    ]
  },
  {
    name: "周边定制",
    title: "创建定制商品",
    description: "AI工具可以轻松地为按需印刷的产品（如t恤、马克杯和海报）设计定制图案、图形和艺术品，允许创作者在没有丰富的平面设计经验的情况下快速生成有市场的设计。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/print-on-demand.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/print-on-demand.png"
    ]
  },
  {
    name: "产品摄影",
    title: "专业产品展示",
    description: "试用背景替换，照明增强和构图工具将基本的产品照片转换为专业的营销海报，而无需昂贵的照片拍摄。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/product-photography.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/product-photography.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/product-photography.png"
    ]
  },
  {
    name: "游戏美工",
    title: "创造游戏美术资源",
    description: "游戏开发者可以通过使用我们的工具来生成纹理、环境元素、角色概念和UI组件来加速制作。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/game-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/game-design.png"
    ]
  },
  {
    name: "室内设计",
    title: "可视化室内空间",
    description: "室内设计师可以使用工具来可视化房间的变化，测试不同的配色方案、家具布置和设计元素。创建逼真的渲染空间从简单的草图或现有的照片。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/room-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/room-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/room-design.png"
    ]
  },
  {
    name: "营销设计",
    title: "为营销提供视觉素材",
    description: "营销人员可以使用工具快速创建专业的社交媒体帖子、广告、横幅和促销材料。",
    images: [
      "https://static.dzine.ai/dzine_config/neo-index/use-case/logo-design.png",
      "https://static.dzine.ai/dzine_config/neo-index/use-case/logo-design.png"
    ]
  }
]);

// Reset active image index when tab changes
watch(activeTab, () => {
  activeImageIndex.value = 0;
});

const setActiveTab = (index) => {
  activeTab.value = index;
};

const setActiveImage = (index) => {
  activeImageIndex.value = index;
};

const nextImage = () => {
  if (activeImageIndex.value < tabs.value[activeTab.value].images.length - 1) {
    activeImageIndex.value++;
  }
};

const prevImage = () => {
  if (activeImageIndex.value > 0) {
    activeImageIndex.value--;
  }
};

// Lifecycle hooks
onMounted(() => {
  // Show promo popup after 3 seconds
  setTimeout(() => {
    showPromoPopup.value = true;
  }, 3000);
});
</script>

<style>
:root {
  --color-background: #000000;
  --color-foreground: #ffffff;
  --color-primary: #f5c518;
  --color-primary-hover: #e6b400;
  --color-secondary: #1a1a1a;
  --color-secondary-hover: #2a2a2a;
  --color-muted: #666666;
  --color-border: #333333;
  --color-green-bg: linear-gradient(to bottom right, #0d3320, #051a10);
  --color-amber-bg: linear-gradient(to bottom right, #3a2e00, #1a1500);
  --color-purple-bg: linear-gradient(to bottom right, #2e0d3a, #15051a);
  --color-blue-bg: linear-gradient(to bottom right, #0d1a3a, #05091a);
  --border-radius: 8px;
  --transition: all 0.3s ease;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

html,
body {
  height: 100%;
  overflow-x: hidden;
  overflow-y: auto;
}

body {
  background-color: var(--color-background);
  color: var(--color-foreground);
  line-height: 1.6;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
}

.container {
  width: 100%;
  margin: 0 auto;
  padding: 0 2rem;
}

/* Navigation */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  background-color: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(10px);
  z-index: 100;
}

.navbar-left,
.navbar-right {
  display: flex;
  align-items: center;
}

.logo {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--color-foreground);
  text-decoration: none;
  margin-right: 2rem;
}

.beta {
  font-size: 0.6rem;
  vertical-align: super;
  color: var(--color-muted);
  margin-left: 2px;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-link {
  color: #fff;
  text-decoration: none;
  transition: var(--transition);
  margin-right: 2rem;
  font-size: 1.2rem;
}

.nav-link:hover {
  color: var(--color-foreground);
}

.dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-btn {
  background: none;
  border: none;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: var(--transition);
}

.dropdown-btn:hover {
  color: var(--color-foreground);
}

.dropdown-arrow {
  font-size: 0.6rem;
}

.btn-primary {
  background-color: var(--color-primary);
  color: #000;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: var(--border-radius);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
}

.btn-primary:hover {
  background-color: var(--color-primary-hover);
}

.btn-secondary {
  background-color: var(--color-secondary);
  color: var(--color-foreground);
  border: 1px solid var(--color-border);
  padding: 0.5rem 1rem;
  border-radius: var(--border-radius);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
}

.btn-secondary:hover {
  background-color: var(--color-secondary-hover);
}

.btn-outline {
  background-color: transparent;
  color: var(--color-foreground);
  border: 1px solid var(--color-border);
  padding: 0.5rem 1rem;
  border-radius: var(--border-radius);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
}

.btn-outline:hover {
  background-color: var(--color-secondary);
}

.btn-large {
  padding: 0.75rem 1.5rem;
  font-size: 1.1rem;
}

.btn-full {
  width: 100%;
}

.mobile-menu-btn {
  display: none;
  background: none;
  border: none;
  cursor: pointer;
}

.mobile-menu-btn span {
  display: block;
  width: 25px;
  height: 2px;
  background-color: var(--color-foreground);
  margin: 5px 0;
  transition: var(--transition);
}

.mobile-menu {
  display: none;
  position: fixed;
  top: 60px;
  left: 0;
  right: 0;
  background-color: var(--color-background);
  padding: 1rem;
  border-bottom: 1px solid var(--color-border);
  flex-direction: column;
  gap: 1rem;
  transform: translateY(-100%);
  transition: transform 0.3s ease;
  z-index: 99;
}

.mobile-menu.open {
  transform: translateY(0);
}

.mobile-link {
  color: var(--color-muted);
  text-decoration: none;
  padding: 0.5rem 0;
  transition: var(--transition);
}

.mobile-link:hover {
  color: var(--color-foreground);
}

.mobile-btn {
  margin-top: 0.5rem;
}

/* Hero Section */







.bg-green {
  background: var(--color-green-bg);
}

.bg-amber {
  background: var(--color-amber-bg);
}

.bg-purple {
  background: var(--color-purple-bg);
}

.bg-blue {
  background: var(--color-blue-bg);
}


.tool-icon {
  width: 16px;
  height: 16px;
}

.tool-title {
  font-weight: 500;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 2rem;
}

/* Hero Section */
.hero {
  padding: 8rem 0 5rem;
  text-align: center;
  background-color: #000000;
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  
  pointer-events: none;
}

.hero-title {
  font-size: 4rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  line-height: 1.2;
  color: #ffffff;
  letter-spacing: 2px
}

.hero-subtitle {
  font-size: 1.5rem;
  color: #9e9e9e;
  max-width: 800px;
  margin: 0 auto 3rem;
}

.hero-button {
  background-color: #f5c518;
  color: #000000;
  border: none;
  padding: 1rem 2.5rem;
  font-size: 1.25rem;
  font-weight: 600;
  border-radius: 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.hero-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(245, 197, 24, 0.3);
}

.hero-button strong {
  font-weight: 800;
}

/* Tools Grid */
.tools-grid {
  padding: 3rem 0 6rem;
  background-color: #000000;
  position: relative;
  overflow: hidden;
}

.tools-grid::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(ellipse at center, rgba(245, 197, 24, 0.1) 0%, rgba(0, 0, 0, 0) 70%);
  pointer-events: none;
}

.tools-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.tool-card {
  border-radius: 16px;
  padding: 1.5rem;
  height: 350px;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
  transition: transform 0.3s ease;
}

.tool-card:hover {
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

.tool-insert-object {
  background: linear-gradient(135deg, #002a1c 0%, #00402b 100%);
}

.tool-ai-eraser {
  background: linear-gradient(135deg, #002940 0%, #003e66 100%);
}

.tool-expand {
  background: linear-gradient(135deg, #29004d 0%, #3d0073 100%);
}

.tool-enhance {
  background: linear-gradient(135deg, #002a1c 0%, #00402b 100%);
}

.tool-header {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
}

.tool-icon {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 0.75rem;
}

.tool-name {
  font-size: 1.25rem;
  font-weight: 600;
  color: #ffffff;
}

.tool-content {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Demo Styles */
.demo-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.before-image, .after-image, .text-prompt, .generated-image {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
}

.tag {
  position: absolute;
  top: 8px;
  left: 8px;
  background-color: rgba(0, 0, 0, 0.6);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  z-index: 1;
}

.transform-arrow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: rgba(255, 255, 255, 0.8);
}

.text-prompt {
  background-color: rgba(0, 0, 0, 0.2);
  padding: 1rem;
  border-radius: 8px;
  text-align: left;
}

.text-prompt p {
  margin: 0;
  color: white;
  font-size: 0.9rem;
  line-height: 1.4;
}

.character-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
}

.video-demo {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.video-preview {
  position: relative;
  margin-bottom: 1rem;
}

.play-button {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  cursor: pointer;
}

.video-frames {
  border-radius: 8px;
  overflow: hidden;
}

.eraser-demo {
  position: relative;
}

.eraser-circle {
  position: absolute;
  top: 30%;
  left: 50%;
  width: 60px;
  height: 60px;
  border: 2px dashed white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.expand-demo {
  position: relative;
}

.expand-arrows {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.arrow {
  background-color: rgba(0, 0, 0, 0.5);
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.corner {
  position: absolute;
  width: 20px;
  height: 20px;
  border: 2px solid white;
}

.top-left {
  top: 20px;
  left: 20px;
  border-right: none;
  border-bottom: none;
}

.top-right {
  top: 20px;
  right: 20px;
  border-left: none;
  border-bottom: none;
}

.bottom-left {
  bottom: 20px;
  left: 20px;
  border-right: none;
  border-top: none;
}

.bottom-right {
  bottom: 20px;
  right: 20px;
  border-left: none;
  border-top: none;
}

.split-image {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
}

.split-line {
  position: absolute;
  top: 0;
  left: 50%;
  width: 2px;
  height: 100%;
  background-color: white;
}

.next-button {
  position: absolute;
  right: 2rem;
  bottom: 50%;
  transform: translateY(50%);
  width: 48px;
  height: 48px;
  background-color: #f5c518;
  color: #000000;
  border: none;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 10;
}

.next-button:hover {
  transform: translateY(50%) scale(1.1);
}

.small-object {
  position: absolute;
  top: -30px;
  left: 50%;
  transform: translate(-50%, 0);
  z-index: 1;
}

.inserted-object {
  position: relative;
}

/* Responsive Adjustments */
@media (max-width: 1200px) {
  .tools-container {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 992px) {
  .hero-title {
    font-size: 3.5rem;
  }
  
  .tools-container {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 3rem;
  }
  
  .hero-subtitle {
    font-size: 1.25rem;
  }
  
  .tools-container {
    grid-template-columns: 1fr;
  }
  
  .tool-card {
    height: 300px;
  }
}

@media (max-width: 576px) {
  .hero {
    padding: 6rem 0 4rem;
  }
  
  .hero-title {
    font-size: 2.5rem;
  }
  
  .hero-subtitle {
    font-size: 1.1rem;
  }
  
  .hero-button {
    padding: 0.875rem 1.75rem;
    font-size: 1.1rem;
  }
}

.tool-images {
  position: relative;
}

.before-image {
  width: 112px;
  height: 112px;
}

.before-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
}

.after-image {
  position: absolute;
  right: -16px;
  bottom: -16px;
  width: 112px;
  height: 112px;
}

.after-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
}

.arrow-icon {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
}

/* Features Section */
.features {
  padding: 4rem 2rem;
  background-color: var(--color-secondary);
  text-align: center;
}

.section-title {
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 2rem;
  color: var(--color-foreground);
}

.section-subtitle {
  font-size: 1.25rem;
  color: var(--color-muted);
  margin-bottom: 3rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.feature-card {
  padding: 2rem;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: var(--border-radius);
  transition: var(--transition);
}

.feature-card:hover {
  transform: translateY(-5px);
  background-color: rgba(255, 255, 255, 0.08);
}

.feature-icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(245, 197, 24, 0.1);
  border-radius: 50%;
  color: var(--color-primary);
}

.feature-title {
  font-size: 1.25rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

.feature-description {
  color: var(--color-muted);
}

/* Pricing Section */
.pricing {
  padding: 4rem 2rem;
  text-align: center;
}

.pricing-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.pricing-card {
  padding: 2rem;
  background-color: var(--color-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--color-border);
  transition: var(--transition);
  position: relative;
}

.pricing-card.featured {
  border-color: var(--color-primary);
  transform: scale(1.05);
}

.pricing-badge {
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  background-color: var(--color-primary);
  color: #000;
  padding: 0.25rem 1rem;
  border-radius: 20px;
  font-weight: 500;
  font-size: 0.875rem;
}

.pricing-header {
  margin-bottom: 2rem;
}

.pricing-title {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

.pricing-price {
  font-size: 2.5rem;
  font-weight: bold;
}

.period {
  font-size: 1rem;
  color: var(--color-muted);
  font-weight: normal;
}

.pricing-features {
  list-style: none;
  margin-bottom: 2rem;
  text-align: left;
}

.pricing-features li {
  padding: 0.5rem 0;
  color: var(--color-muted);
}

.pricing-features li::before {
  content: "✓";
  color: var(--color-primary);
  margin-right: 0.5rem;
}

/* Testimonials */
.testimonials {
  padding: 4rem 2rem;
  background-color: var(--color-secondary);
  text-align: center;
}

.testimonials-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.testimonial-card {
  padding: 2rem;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: var(--border-radius);
  text-align: left;
}

.testimonial-content {
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
  font-style: italic;
}

.testimonial-author {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.testimonial-avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background-color: var(--color-muted);
}

.testimonial-name {
  font-weight: 500;
}

.testimonial-role {
  color: var(--color-muted);
  font-size: 0.875rem;
}

/* CTA Section */
.cta {
  padding: 6rem 2rem;
  text-align: center;
  max-width: 800px;
  margin: 0 auto;
}

.cta-title {
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

.cta-subtitle {
  font-size: 1.25rem;
  color: var(--color-muted);
  margin-bottom: 2rem;
}

.cta-note {
  margin-top: 1rem;
  color: var(--color-muted);
  font-size: 0.875rem;
}

/* Pre-Footer Section */
.pre-footer {
  background-color: var(--color-background);
  padding: 6rem 2rem;
  text-align: center;
  border-bottom: 1px solid var(--color-border);
}

.pre-footer-title {
  font-size: 3.5rem;
  font-weight: bold;
  margin-bottom: 2.5rem;
  color: var(--color-foreground);
}

/* New Footer */
.new-footer {
  background-color: var(--color-background);
  padding: 4rem 0 2rem;
  color: var(--color-foreground);
}

.footer-top {
  display: grid;
  grid-template-columns: 1fr 3fr;
  gap: 3rem;
  margin-bottom: 3rem;
}

.footer-brand {
  display: flex;
  flex-direction: column;
}

.footer-logo {
  display: inline-flex;
  align-items: baseline;
  text-decoration: none;
  margin-bottom: 1rem;
}

.logo-text {
  font-size: 2.5rem;
  font-weight: bold;
  color: var(--color-foreground);
}

.logo-beta {
  font-size: 0.8rem;
  color: var(--color-primary);
  margin-left: 4px;
}

.footer-tagline {
  color: #ffffffb4;
  margin-bottom: 2rem;
  line-height: 1.5;
  text-align: left;
  font-weight: 200;
}

.footer-social {
  display: flex;
  gap: 1rem;
}

.social-icon {
  color: var(--color-muted);
  transition: var(--transition);
}

.social-icon:hover {
  color: var(--color-foreground);
}

.footer-links-section {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 2rem;
  text-align: left;
}

.footer-column-title {
  font-size: 0.8rem;
  font-weight: bold;
  margin-bottom: 1.5rem;
  color: var(--color-primary);
}

.footer-links {
  list-style: none;
}

.footer-links li {
  margin-bottom: 0.25rem;
}

.footer-links a {
  color: #ffffffb4;
  text-decoration: none;
  font-size: 0.9rem;
  transition: var(--transition);
  font-weight: 100;


}

.footer-links a:hover {
  color: var(--color-foreground);
}

.footer-bottom {
  padding-top: 2rem;
  border-top: 1px solid var(--color-border);
}

.copyright {
  color: var(--color-muted);
  font-size: 0.875rem;
}

/* Login Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  z-index: 1000;
}

.login-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: var(--color-secondary);
  border-radius: var(--border-radius);
  width: 100%;
  max-width: 450px;
  z-index: 1001;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid var(--color-border);
}

.modal-title {
  font-size: 1.25rem;
  font-weight: bold;
}

.modal-close {
  background: none;
  border: none;
  color: var(--color-muted);
  font-size: 1.5rem;
  cursor: pointer;
  transition: var(--transition);
}

.modal-close:hover {
  color: var(--color-foreground);
}

.modal-body {
  padding: 1.5rem;
}

.login-form {
  margin-bottom: 1.5rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.form-group input {
  width: 100%;
  padding: 0.75rem;
  border-radius: var(--border-radius);
  border: 1px solid var(--color-border);
  background-color: rgba(255, 255, 255, 0.05);
  color: var(--color-foreground);
  transition: var(--transition);
}

.form-group input:focus {
  outline: none;
  border-color: var(--color-primary);
  background-color: rgba(255, 255, 255, 0.1);
}

.password-input {
  position: relative;
}

.password-toggle {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: var(--color-muted);
  cursor: pointer;
  transition: var(--transition);
}

.password-toggle:hover {
  color: var(--color-foreground);
}

.form-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.checkbox-container {
  display: flex;
  align-items: center;
  cursor: pointer;
  color: var(--color-muted);
  font-size: 0.875rem;
}

.checkbox-container input {
  margin-right: 0.5rem;
}

.forgot-password {
  color: var(--color-primary);
  text-decoration: none;
  font-size: 0.875rem;
  transition: var(--transition);
}

.forgot-password:hover {
  text-decoration: underline;
}

.divider {
  display: flex;
  align-items: center;
  text-align: center;
  margin: 1.5rem 0;
}

.divider::before,
.divider::after {
  content: '';
  flex: 1;
  border-bottom: 1px solid var(--color-border);
}

.divider span {
  padding: 0 10px;
  color: var(--color-muted);
  font-size: 0.875rem;
}

.social-login {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.btn-social {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem;
  border-radius: var(--border-radius);
  border: 1px solid var(--color-border);
  background-color: transparent;
  color: var(--color-foreground);
  cursor: pointer;
  transition: var(--transition);
}

.btn-social:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.signup-prompt {
  text-align: center;
  color: var(--color-muted);
  font-size: 0.875rem;
}

.signup-prompt a {
  color: var(--color-primary);
  text-decoration: none;
  transition: var(--transition);
}

.signup-prompt a:hover {
  text-decoration: underline;
}

/* Promo Popup */
.promo-popup {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: #fff1f5;
  border-radius: var(--border-radius);
  width: 100%;
  max-width: 400px;
  z-index: 1001;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
}

.promo-close {
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: rgba(255, 255, 255, 0.8);
  border: none;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  cursor: pointer;
  z-index: 10;
  transition: var(--transition);
}

.promo-close:hover {
  background-color: rgba(255, 255, 255, 1);
}

.promo-content {
  padding: 2rem;
  text-align: center;
  position: relative;
}

.promo-decoration {
  position: absolute;
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background-color: rgba(244, 114, 182, 0.2);
}

.top-right {
  top: -50px;
  right: -50px;
}

.bottom-left {
  bottom: -50px;
  left: -50px;
}

.promo-title {
  color: #f472b6;
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.promo-discount {
  color: #f472b6;
  font-size: 4rem;
  font-weight: bold;
  line-height: 1;
}

.discount-number {
  font-size: 4rem;
}

.discount-percent {
  font-size: 2rem;
  vertical-align: super;
}

.discount-label {
  color: #f472b6;
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.promo-subtitle {
  color: #f472b6;
  margin-bottom: 1.5rem;
}

.promo-code {
  background-color: white;
  border-radius: var(--border-radius);
  padding: 0.75rem;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.code-label {
  color: #666;
}

.code {
  border: 1px solid #f8bbd0;
  border-radius: 4px;
  padding: 0.25rem 0.5rem;
  color: #f472b6;
  font-weight: bold;
}

.promo-note {
  color: #f472b6;
  font-size: 0.875rem;
  margin-bottom: 1.5rem;
}

.promo-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

/* SVG Icons */
[class^="svg-"] {
  display: inline-block;
  width: 16px;
  height: 16px;
}

/* Tool Icons */
.svg-image,
.svg-text,
.svg-user,
.svg-edit,
.svg-plus,
.svg-eraser,
.svg-expand,
.svg-wand,
.svg-bag,
.svg-cube {
  width: 16px;
  height: 16px;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  fill: none;
}

/* Responsive Styles */
@media (max-width: 1200px) {


  .footer-links-section {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 992px) {


  .pricing-grid,
  .features-grid,
  .testimonials-grid {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }

  .footer-top {
    grid-template-columns: 1fr;
  }

  .footer-brand {
    margin-bottom: 2rem;
  }

  .footer-links-section {
    grid-template-columns: repeat(2, 1fr);
  }

  .pre-footer-title {
    font-size: 2.5rem;
  }
}

@media (max-width: 768px) {
  .nav-links {
    display: none;
  }

  .mobile-menu-btn {
    display: block;
  }

  .mobile-menu {
    display: flex;
  }




  .section-title {
    font-size: 2rem;
  }

  .pricing-card.featured {
    transform: none;
  }

  .footer-links-section {
    grid-template-columns: 1fr;
  }

  .footer-bottom {
    flex-direction: column;
    gap: 1rem;
  }



  .pre-footer-title {
    font-size: 2rem;
  }
}

@media (max-width: 576px) {
  .navbar {
    padding: 1rem;
  }


  .features,
  .pricing,
  .testimonials,
  .cta,
  .pre-footer {
    padding: 3rem 1rem;
  }



 

  .login-modal {
    width: 90%;
  }

  .promo-popup {
    width: 90%;
  }

  .btn-social {
    font-size: 0.875rem;
  }

  .footer-links-section {
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  .footer-column-title {
    margin-bottom: 1rem;
  }
}

/* Accessibility Improvements */
:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.use-cases-section {
  padding: 5rem 0;
  background-color: #000000;
  color: #ffffff;
}

.container {

  margin: 0 auto;
  padding: 0 5rem;
}

.use-cases-title {
  font-size: 3.5rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 0.5rem;
}

.use-cases-subtitle {
  font-size: 1.25rem;
  color: #9e9e9e;
  text-align: center;
  margin-bottom: 3rem;
}

.use-cases-tabs {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 3rem;
  position: relative;
}

.use-cases-tabs::after {
  content: '';
  position: absolute;
  bottom: -10px;
  left: 0;
  width: 100%;
  height: 1px;
  background-color: #333;
}

.tab-button {
  background: transparent;
  border: none;
  color: #9e9e9e;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: color 0.3s ease;
  position: relative;
}

.tab-button:hover {
  color: #ffffff;
}

.tab-button.active {
  color: #ffffff;
}

.tab-button.active::after {
  content: '';
  position: absolute;
  bottom: -10px;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: var(--color-primary, #f5c518);
  z-index: 1;
}

.use-cases-content {
  display: flex;
  flex-wrap: wrap;
  gap: 3rem;
  align-items: center;
}

.content-left {
  flex: 1;
  min-width: 300px;
}

.content-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
}

.content-description {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #d1d1d1;
}

.content-right {
  flex: 2;
  min-width: 300px;
}

.image-carousel {
  position: relative;
  overflow: hidden;
  border-radius: 12px;
}

.carousel-images {
  display: flex;
  transition: transform 0.5s ease;
}

.carousel-image {
  width: 100%;
  flex-shrink: 0;
  object-fit: cover;
  border-radius: 12px;
}

.carousel-controls {
  position: absolute;
  bottom: 20px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.control-button {
  background-color: rgba(0, 0, 0, 0.6);
  color: #ffffff;
  border: none;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.control-button:hover {
  background-color: rgba(0, 0, 0, 0.8);
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.carousel-dots {
  display: flex;
  gap: 0.5rem;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.4);
  border: none;
  padding: 0;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.dot.active {
  background-color: var(--color-primary, #f5c518);
  transform: scale(1.2);
}

/* Responsive adjustments */
@media (max-width: 992px) {
  .use-cases-content {
    flex-direction: column;
  }

  .content-left {
    text-align: center;
  }

  .use-cases-title {
    font-size: 2.5rem;
  }

  .content-title {
    font-size: 2rem;
  }
}

@media (max-width: 768px) {
  .use-cases-tabs {
    gap: 0.5rem;
  }

  .tab-button {
    font-size: 0.9rem;
    padding: 0.5rem;
  }

  .use-cases-title {
    font-size: 2rem;
  }
}

@media (max-width: 576px) {
  .use-cases-section {
    padding: 3rem 0;
  }

  .use-cases-tabs {
    flex-direction: column;
    gap: 0.25rem;
  }

  .tab-button {
    width: 100%;
    padding: 0.75rem;
    text-align: center;
  }

  .tab-button.active::after {
    height: 1px;
  }
}

.feature-sections {
  width: 100%;
  background-color: #000000;
  color: #ffffff;
}

.feature-section {
  padding: 5rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.feature-container {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 4rem;
}

.feature-section-reverse .feature-container {
  flex-direction: row-reverse;
}

.feature-image {
  flex: 1;
  min-width: 300px;
  border-radius: 16px;
  overflow: hidden;
}

.feature-image img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 16px;
  transition: transform 0.5s ease;
}

.feature-image:hover img {
  transform: scale(1.02);
}

.feature-image-colored {
  background-color: var(--feature-bg-color, #FFF176);
  padding: 2rem;
}

.feature-content {
  flex: 1;
  min-width: 300px;
}

.feature-title {
  font-size: 3rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
}

.feature-description {
  font-size: 1.25rem;
  line-height: 1.6;
  color: #d1d1d1;
  margin-bottom: 2rem;
}

.feature-description strong {
  color: #ffffff;
}

.feature-button {
  display: inline-block;
  background-color: transparent;
  color: #ffffff;
  border: 2px solid #ffffff;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  border-radius: 8px;
  text-decoration: none;
  transition: all 0.3s ease;
}

.feature-button:hover {
  background-color: #ffffff;
  color: #000000;
}

/* Responsive styles */
@media (max-width: 992px) {
  .feature-container {
    flex-direction: column;
    gap: 2rem;
  }

  .feature-section-reverse .feature-container {
    flex-direction: column;
  }

  .feature-content {
    text-align: center;
  }

  .feature-title {
    font-size: 2.5rem;
  }
}

@media (max-width: 768px) {
  .feature-section {
    padding: 3rem 1.5rem;
  }

  .feature-title {
    font-size: 2rem;
  }

  .feature-description {
    font-size: 1.1rem;
  }
}

@media (max-width: 576px) {
  .feature-section {
    padding: 2.5rem 1rem;
  }

  .feature-title {
    font-size: 1.75rem;
  }

  .feature-description {
    font-size: 1rem;
  }

  .feature-button {
    width: 100%;
    text-align: center;
  }
}
</style>



