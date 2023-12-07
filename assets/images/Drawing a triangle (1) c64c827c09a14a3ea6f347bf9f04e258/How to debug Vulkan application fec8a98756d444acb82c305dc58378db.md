# How to debug Vulkan application

By default, the validation layers will print debug messages to the standard output. We can also handle them ourselves by providing an explicit callback to choose the level of severity of the message we want to see.

To do so, we need to add debugging support in createInstance() and also setupDebugMessenger().

### createInstance()

**Debugging vkCreateInstance and vkDestroyInstance**

Later we will introduce how to setup debug messenger to the whole program. For debugging instance creation and destruction, we simply pass a pointer to a VkDebugUtilsMessengerCreateInfoEXT struct in the pNext of VkInstanceCreateInfo. The debugCreateInfo will be the same as the later part.

```cpp
createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
```

VkDebugUtilsMessengerCreateInfoEXT struct required us to set the messageSeverity, messageType and pointer to user callback function.

The signature of the callback function for Vulkan to call is

```cpp
VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                       VkDebugUtilsMessageTypeFlagsEXT messageType,
                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                       void* pUserData)
```

### setupDebugMessenger()

We will need VkDebugUtilsMessengerCreateInfoEXT struct and the instance to create a VkDebugUtilsMessengerEXT object, our debug messenger.

vkCreateDebugUtilsMessengerEXT and vkDestroyDebugUtilsMessengerEXT is an extension, so we need to manually find the address of the function. We will use vkGetInstanceProcAddr to get the pointer to function.

```cpp
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
```