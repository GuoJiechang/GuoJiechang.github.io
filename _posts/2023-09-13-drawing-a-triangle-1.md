---
layout: post
title: Drawing a triangle using Vulkan (1)
date: 2023-09-13 14:33 -0500
categories: CG_Notes
tag: CG
---
After writing or copying over 1000 lines of code, I finally rendered my first triangle using Vulkan. Seeing it on the screen was so exciting!

![Untitled](/assets/images/Drawing%20a%20triangle%20(1)%20c64c827c09a14a3ea6f347bf9f04e258/Untitled.png)

I want to learn Vulkan to study state-of-the-art techniques in the field of computer graphics. I find the process of learning new techniques enjoyable, which makes me feel refreshed and positive. 

## Application outline

The base code is simple. Basically four steps involved in an application:

```cpp
void HelloTriangleApplication::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanUp();
}
```

### initWindow()

One major difference between OpenGL and Vulkan is that, for OpenGL, a window is required to create a context. If one would like to do off-screen rendering, a hidden window will be the solution. In contrast, Vulkan can function perfectly well without a window. 

We still need to know how to show the rendered object in a window. We use GLFW library to create a window to display the rendered image on the screen. 

### initVulkan()

Initializing Vulkan is the most verbose part of the code. Every detail related to the graphics API needs to be set up explicitly from scratch in the application. As you can see in the following code, it takes 13 steps to initialize Vulkan. We will go into detail on each part and try to understand them.

```cpp
void HelloTriangleApplication::initVulkan(){
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
}
```

### mainLoop()

In the main loop, we will draw a frame until the window is closed or we encountered an error.

Everything in the drawFrame() are asynchronous, when we exit the loop, the operations may still be going on, thus, we need to wait for the device to finish all the operations before exiting mainLoop() and enter cleanUp().

```cpp
void HelloTriangleApplication::mainLoop(){
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
    
    vkDeviceWaitIdle(device);
}
```

### cleanUp()

Once the window is closed, everything we created need to be destroy explicitly.

## Vulkan Initialization

### createInstance()

VkInstance instance is needed for every Vulkan application which is the connection between the application and the Vulkan library. VkInstance is the first Vulkan object we created.

The general pattern to create a Vulkan object is:

- Pointer to struct with creation info, a struct related to the object we want to create
- Pointer to custom allocator callbacks, for now, idk what it is
- Pointer to the variable that stores the handle to the new object
- The function will return VkResult to tell if we create the object successfully

Vulkan object created by us needs to be explicitly destroyed. vkCreateXXX + vkDestroyXXX, vkAllocateXXX + vkFreeXXX

The VkInstanceCreateInfo createInfo contains the information about the application, global extension, and enabled layers like the validation layer for debugging purpose. Call vkCreateInstance to create the Vulkan instance.

```cpp
VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
```

The correspondence destroy function is vkDestroyInstance(instance,nullptr).

Now let’s go into detail on the vkInstanceCreateInfo. Typically, Vulkan struct will need us to explicitly set the VkStructureType. The information about the application including the name of the application, version information, etc. The enabled layers information contains the layers we want to enable. Vulkan introduces validation layers system to help us debug our program. The enabled extension information contains the extensions we enabled in the program.

```cpp
typedef struct VkInstanceCreateInfo {
    VkStructureType             sType; //the type of the struct need to be specified
    const void*                 pNext; //optional
    VkInstanceCreateFlags       flags;

		//application information
    const VkApplicationInfo*    pApplicationInfo;
		//enable layers information
    uint32_t                    enabledLayerCount;
    const char* const*          ppEnabledLayerNames;
		//enable extension information
    uint32_t                    enabledExtensionCount;
    const char* const*          ppEnabledExtensionNames;
} VkInstanceCreateInfo;
```

**Extensions**

1. glfwExtensions
    
    Vulkan is a platform agnostic API, extensions are required to interface with the window system. GLFW provide built-in function that returns the required extensions.
    
    ```cpp
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    ```
    
2. debug extensions
    
    To setup a debug messenger with a callback we need to enable the extension: VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    

 3.  solve MacOS issue VK_ERROR_INCOMPATIBLE_DRIVER

VK_KHR_PORTABILITY_subset extension is mandatory

(1) VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME

Also, a flag VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR should be added to createInfo

(2) "VK_KHR_get_physical_device_properties2”

Note that the first extension is a macro definition, the second is a string

**Validation** **Layers**

We enabled "VK_LAYER_KHRONOS_validation” layer if in Debug mode.

We also need to check if the system support validation layer.

The general pattern to get properties from Vulkan is list in the following code. We will call the vkEnumerateXXXProperties for two times. The first time is to get the number of the properties, then we can initialize the container for handling the exact data. 

```cpp
uint32_t count = 0;
//the first parameter
vkEnumerateXXXProperties(&count,nullptr);
std::vector<VkXXXProperties> properties(count);
vkEnumerateXXXProperties(&count,properties.data());
```

For checking the validation layer VkLayerProperties support, the function is vkEnumerateInstanceLayerProperties.


### createSurface()

After creating the instance, we need to create a VkSurfaceKHR object to present rendered images to the screen. Here we use the built-in function of GLFW to create window surface.

```cpp
glfwCreateWindowSurface(instance, window, nullptr, &surface)
```

### pickPhysicalDevice()

To select a suitable physical device which is a graphics card in the system. We are able to select any number of graphics cards and use them simultaneously. For beginner, we only use the first graphic card ): looks like I have choice, developing on MacOS.

To get available physical devices VkPhysicalDevice, is similar to get properties except we need the instance. The function is 

```cpp
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
```

From the available physical devices, we will choose the one suitable for our application.

**isDeviceSuitable(device)**

The suitable device will satisfy the following three requirements: queue families, extensions, swap chain

1. findQueueFamilies(device)

Most operations performed with Vulkan from draw commands to uploading textures requires commands to be submitted to a queue. Different types of queues are from different queue families which support different subset of commands. We need to find out the queue families that are supported by the device as well as meet our requirements. In this simply triangle application, we need the indices of two queue families includes graphics family that supports graphics commands and present family that supports presenting the rendered image to the surface. To get the supported queue families VkQueueFamilyProperties is similar to get properties as mentioned before.  

```cpp
vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data())
```

Then we iterate over the queue families we get and find out the index of queue we needed.

1. checkDeviceExtensionSupport(device)

We first get all the available extensions, then iterate over the extensions to check if the required extensions exist. Currently, the required device extension is VK_KHR_SWAPCHAIN_EXTENSION_NAME

```cpp
vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
```

1. check swap chain support

Swap chain is like a “default framebuffer” to store the images waiting to be presented to the screen. If swap chain extension in 2 is supported, we need to query the detail of the swap chain. And check if it meet our requirement. In this simple program, we will consider the swap chain adequate if it supports at least  one surface format(pixel format, color space) and one presentation mode. 

If we find the queue families indices and all the extensions supported, we will choose this device as our physical device.

### createLogicalDevice()

A logical device is needed to set up to interface with the physical device we just created where we can describe more specifically how we will use the physical device.

We need to specifying the queues to be created, the device features to be used, the extensions to be enabled, and the validation layer to be enabled.

To solve the issue with VK_ERROR_INCOMPATIBLE_DRIVER on MacOS, we need to enable “VK_KHR_portability_subset” extension.

The we based on the above information included in VkDeviceCreateInfo struct, we create the logical device.

The last step is to retrieve the queue handle we created.

### createSwapChain()

When picking physical device, we already enable the swap chain support and query its detail, now it’s time for us to actually create swap chain.

We will choose the properties of the swap chain including: surface format(pixel format, color space), presentation mode, swap extent, image count in a swap chain.

After we choose the ideal value of the properties, we will create the swap chain object.

I’ll keep the detail of the properties selection for now. It is better for me to get the bigger picture first.

### createImageViews()

A VkImageView describes how to view an image. We will create VkImageView that match the image in the swap chain. That is to say we will create a VkImageView object for every image in the swap chain. In this simple demo, we simply treat the images as 2D textures.

### createRenderPass()

Render pass object wrap the information for the framebuffer attachments. Including the number of color and depth buffers that will be used while rendering, number of samples for each buffer, etc. For this demo, a single color buffer attachment and a subpass were created for the render pass object.