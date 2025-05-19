import Vue from 'vue'
import axios from 'axios';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import App from './App.vue'
import router from './router'
import store from './store'
import BaiduMap from 'vue-baidu-map'
import '@/utils/rem.js'


// 设置默认配置（可选）
axios.defaults.baseURL = '/rtsi/'; // 设置 API 基础 URL
Vue.prototype.$http = axios;
// 创建并使用一个自定义实例（可选）
const request = axios.create({
  headers: {'Content-Type': 'application/json'},
  // 其他自定义配置项...
});
Vue.prototype.$api = request; // 如果需要区分不同类型的请求，可以创建多个实例并分别挂载

Vue.use(ElementUI);

Vue.use(BaiduMap, {
    ak: 'FaRpbk6I6i7084DcIRoQNCv20WpHWfV7'
})

Vue.config.productionTip = false

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
