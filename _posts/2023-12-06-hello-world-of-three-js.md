---
layout: post
title: Hello World of Three.js
date: 2023-12-06 15:48 -0600
---
This is the demo I created when learning Three.js.
{% include threejshelloworld1.html %}

## three.js with jekyll post
I finally figured out how to embed HTML with three.js in a markdown file.
The HTML file looks like this. Created a js folder inside assets, and put the javascript file inside it.
```
<style>
.threejs {
    position: relative;
    width: 100%;
    padding-top: 56.25%; /* 16:9 aspect ratio */
  }
  .threejs > * {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
  }
</style>
<div class='threejs'>
    <div id='cube'></div>
</div>
<script type = 'module' src='/assets/js/helloworldthreejs1.js'></script>
```

It's not the optimal solution I think, because I didn't use npm to install Three.js in this blog project. Instead, simply use the following code inside the 'helloworldthreejs1.js' file
```
    import * as THREE from 'https://unpkg.com/three@0.126.1/build/three.module.js'
```

## hello spinning cubes
A common three.js application is to pass a Scene and a Camera to a Renderer and it renders the portion of the 3D scene that is inside the frustum of the camera as a 2D image to a canvas.
Here is the structure of this demo. The demo is adapted from https://threejs.org/manual/#en/fundamentals.
![Untitled](/assets/images/threejs-3cubes-scene.svg)
We have three main components in this program, Renderer, Camera, and Scene. We also need a canvas to put our rendered 2D image.

### get container to put the rendered 2D image
The name of the container should be the same as the one in the HTML file.
```
    var container = document.getElementById("cube");
    var width = container.clientWidth;
    var height = container.clientHeight;
```

### create scene
Our scene contains 3 cubes with phong materials as well as a directional light source. Let's create the scene first.
```
    const scene = new THREE.Scene();
```

Create three cube instances using the same geometry, save material and different colors. And add each cube to the scene.

```
const geometry = new THREE.BoxGeometry( 1, 1, 1 );

function makeInstance(geometry, color, x) {
	const material = new THREE.MeshPhongMaterial({color});
   
	const cube = new THREE.Mesh(geometry, material);
	scene.add(cube);
   
	cube.position.x = x;
   
	return cube;
}

//cube instances
const cubes = [
	makeInstance(geometry, 0x44aa88,  0),
	makeInstance(geometry, 0x8844aa, -2),
	makeInstance(geometry, 0xaa8844,  2),
];
```

Create light source to make the cube 3D. And add light source to the scene.
```
//light source
const color = 0xFFFFFF;
const intensity = 3;
const light = new THREE.DirectionalLight(color, intensity);
light.position.set(-1, 2, 4);
scene.add(light);
```

### create camera
```
const camera = new THREE.PerspectiveCamera( 75, width / height, 0.1, 1000 );
camera.position.z = 5;
```

### create renderer and add to container
We use WebGLRenderer here.
```
const renderer = new THREE.WebGLRenderer();
renderer.setSize(width, height);
container.appendChild(renderer.domElement);
```

### Loop the scene
```
function animate() {
	//Requests the browser to call the animate function before the next repaint, creating a smooth animation loop.
	requestAnimationFrame( animate );
	
	cubes.forEach((cube, ndx) => {
		cube.rotation.x += 0.01;
		cube.rotation.y += 0.01;
	  });

	renderer.render( scene, camera );
}

animate();
```










