import Vue from 'vue'
import VueRouter from 'vue-router'
import HomeView from '../views/HomeView.vue'
import index from '@/pages/index.vue';
import dashboard from '@/pages/dashboard.vue'
import tool from '@/pages/tool.vue'
import cn from '@/pages/index_cn.vue'
import dashboard_cn from '@/pages/dashboard_cn.vue'




Vue.use(VueRouter)

const routes = [

  
  {
    path: '/',
    name: 'index',
    component: index
  },

  {
    path: '/dashboard',
    name: 'dashboard',
    component: dashboard
  },
  {
    path: '/dashboard/cn',
    name: 'dashboard_cn',
    component: dashboard_cn
  },
  {
    path: '/tool',
    name: 'tool',
    component:tool
  },
  {
    path: '/cn',
    name: 'cn',
    component:cn
  },
  


]

const router = new VueRouter({
  routes
})

export default router
