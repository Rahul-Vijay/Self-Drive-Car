# Self-Drive-Car
## 2-D Self driving car with Reinforcement learning 

### This project makes use of the following technologies:-

* Python 2.7.x
* Kivy
* PyTorch
* And a bunch of other python modules

### To run this project, follow these steps:
1. Install Anaconda(for beginners) or Python 2.7.x from [here](https://www.python.org/downloads/) 
2. Set up a virtual environment. Follow [here](https://virtualenv.pypa.io/en/stable/userguide/#usage) for more help
3. Next activate the virtual env by -
	* Linux/Mac - ``` source yourFolder/bin/activate ```
4. Install the required modules listed in __requirements.txt__
5. Next get Kivy for your machine (follow link for instructions) -
	* [Windows](https://kivy.org/docs/installation/installation-windows.html)
	* [macOS/OSX](https://kivy.org/docs/installation/installation-osx.html)
	* [Linux](https://kivy.org/docs/installation/installation-linux.html)
 6. Get [Pytorch](https://pytorch.org/) according to your machine's configuration
 	* For my machine (macOS) without CUDA this command was run-
	```pip install torch torchvision``` <br>
 
 __NOTE:__ For macOS 10.12 or later do (do not sudo brew, you can sudo pip) -
 ```sh
 brew install pkg-config sdl2 sdl2_image sdl2_ttf sdl2_mixer gstreamer
 pip install Cython==0.27
 pip install https://github.com/kivy/kivy/archive/master.zip
 ```
 There are some issues regarding the stable Kivy with python 2 and Cython's latest stable version. <br>
 Recommended use - Dev version kivy (stable with python 2.7.x and cython earlier version)


NOTE: The use of PyTorch over TensorFlow is because it computes dynamic graph calculations faster
