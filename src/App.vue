<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<style lang="scss">
body{
  margin: 0;
  background: #f1f6fd;
  height: 100vh;
  overflow: hidden;
}
/**
@font-face {
  font-family: 'Alibaba PuHuiTi';
  src: url('../../road_sys/src/assets/AlibabaPuHuiTi-3/AlibabaPuHuiTi-3-35-Thin/AlibabaPuHuiTi-3-35-Thin.ttf');
} */
#app {
  /**font-family: Avenir, Helvetica, Arial, sans-serif;**/
  //font-family: 'Alibaba PuHuiTi';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  overflow-y: scroll;
  color: #fff;
  
}
button{
  //font-family: 'Alibaba PuHuiTi';
  cursor: pointer;
  border: 0;
  font-weight: 700;
  //font-size: 16px;
  color: #FFFFFF;
}

input{
  border:0;
}

input:focus-visible{
  border: 0;
  outline:0;
}

nav {
  padding: 0px;

  a {
    font-weight: bold;
    color: #fff;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>
