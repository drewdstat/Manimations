# Explanation of How I Got Started with Manim

I tried to add the 'manim' python package on my own machine by 'pip install'ing it to a Python 3.12 conda environment, but any time I tried to render Latex text, I would encounter an error. I couldn't get TinyTex or any other interpreter to work with manim, even after updating my PATH file path.

As a solution, I instead downloaded the Docker software. Then I followed the instructions on this site to get a JupyterLab instance started in my web browser: https://docs.manim.community/en/stable/installation/docker.html#running-jupyterlab-via-docker.

Once I had downloaded the Docker extension in VS Code, I used the terminal to run the following: docker run -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0. The terminal text then prints out a couple URLs. I opened the second one (starting with "127.") in my web browser.

All code you see in the .py files in the python subfolder of this repo I ran through JupyterLab via Docker on my web browser. The only differences is that the two lines defining a command for the command line (command1 = ...) and the line to run that command (os.system(command1)) were not
what I used in JupyterLab. Instead, I ran %manim SCENENAME in a code chunk, where SCENENAME is replaced with the name of a scene (e.g., PermTest2D). Replace those aforementioned two lines with the %manim line in the Docker JupyterLab.

I then downloaded the resulting .mp4 file from JupyterLab after using the file directory explorer on the left side to find the output files. Keep in mind that once you close this Docker instance down, all files are lost, so you'll want to copy/download them and save them to your local machine. 

More instruction for installing and running manim through Docker's JupyterLab can be found in this YouTube video: https://www.youtube.com/watch?v=S7DvtP20ggU&t=111s.
