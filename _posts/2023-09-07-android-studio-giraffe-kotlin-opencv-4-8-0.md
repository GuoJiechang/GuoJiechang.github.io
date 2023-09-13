---
layout: post
title: Android Studio Giraffe + Kotlin + OpenCV 4.8.0
date: 2023-09-07 13:22 -0500
---
After hours of attempting, I finally integrated OpenCV 4.8.0 to Android Studio Giraffe with the Kotlin project.

I’ll skip the steps for downloading AS and OpenCV. Just remember to download the Android version of OpenCV and extract the zipped file to whatever place you want.

## Create a new “Empty Activity” project

I want to use Compose to develop the UI of the app. Currently, I only learned this from Android Studio's official tutorial. So just create an “Empty Activity”.

File→New→New Project→Empty Activity→Next

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled.png)

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%201.png)

Remember to change the “Build configuration Language” from default to “Groovy DSL(build.gradle)”

Then click “Finish”

## Import OpenCV SDK as a new module

File→New→Import Module→

Select the path to the place where you extracted the OpenCV SDK and change the module name to “OpenCV”

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%202.png)

Then click Finish

Now, we will start our troubleshooting journey.

## Troubleshooting 1: Namespace not specified

```cpp
Namespace not specified. Specify a namespace in the module's build file. See https://d.android.com/r/tools/upgrade-assistant/set-namespace for information about setting the namespace.

If you've specified the package attribute in the source AndroidManifest.xml, you can use the AGP Upgrade Assistant to migrate to the namespace value in the build file. Refer to https://d.android.com/r/tools/upgrade-assistant/agp-upgrade-assistant for general information about using the AGP Upgrade Assistant.
```

Please check this link for more information related to this issue [https://github.com/opencv/opencv/pull/23447](https://github.com/opencv/opencv/pull/23447)

To fix this issue

Change the project view to “Project” where you can find the imported “opencv” module. Inside the “opencv” folder, open “build.gradle”, add “namespace ‘org.opencv’ to the file.

Then sync the gradle by click Try Again on the top of this gradle file.

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%203.png)

Then this problem should be fix. BUILD SUCCESSFUL( which is obviously not yet).

## Add opencv as dependency as the app

File→Project Structure→Dependencies(tab)→app→+→3 Module Dependency→

select opencv module as dependencies→OK→Apply

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%204.png)

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%205.png)

Run the app and lets fix some compile error of OpenCV

## Troubleshooting2: Execution failed for task ':opencv:compileDebugKotlin’

```cpp
Execution failed for task ':opencv:compileDebugKotlin'.
> 'compileDebugJavaWithJavac' task (current target is 1.8) and 'compileDebugKotlin' task (current target is 17) jvm target compatibility should be set to the same Java version.
  Consider using JVM toolchain: https://kotl.in/gradle/jvm/toolchain
```

Open build.gradle of opencv again

Add the following code to the end of the android{} section

```groovy
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(8)
	  }
}

buildFeatures {
    aidl true
}
```

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%206.png)

And click Sync Now. BUILD SUCCESSFUL (not yet)

Run the app again.

## Troubleshooting 3:  Cannot find symbol

```cpp
error: cannot find symbol
import org.opencv.BuildConfig;
^
symbol:   class BuildConfig
location: package org.opencv
```

Add one more buildFeature 

```groovy
buildFeatures **{**    
	aidl true    
	buildConfig true
**}**
```

Now the app should be able to run. 

Congrats!

## Test OpenCV is ready to use

import opencv dependencies, add the OpenCVLoader.initDebug() in the onCreate fun.

```kotlin
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if(OpenCVLoader.initDebug())
        {
            Log.d("OPENCV","Opencv init")
            println("opencv version: ${Core.VERSION}")
        }
        else
        {
            Log.d("OPENCV","Opencv init failed")
        }
        setContent {
            OpenCVIntegrationFinalTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Greeting("Android")
                }
            }
        }
    }
}
```

We can check the logcat

![Untitled](/assets/images/Android%20Studio%20Giraffe%20+%20Kotlin%20+%20OpenCV%204%208%200%2050172eecc0d14bdfaf39049fe49db4a1/Untitled%207.png)

It’s working! Finally!

Reference:
[https://medium.com/@ankitsachan/android-studio-kotlin-open-cv-16d75f8d9969](https://medium.com/@ankitsachan/android-studio-kotlin-open-cv-16d75f8d9969)
[https://forum.opencv.org/t/an-exercise-in-frustration/12964/3?u=ankit_sachan&source=post_page-----16d75f8d9969--------------------------------](https://forum.opencv.org/t/an-exercise-in-frustration/12964/3?u=ankit_sachan&source=post_page-----16d75f8d9969--------------------------------)

