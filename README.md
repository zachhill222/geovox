## INSTALATION
upgrade to python3.11
	$apt install python3.11-dev python3.11-full

1) In the terminal, navigate to the directory where geovox is located.

2) Make a virtual environment
	<username>:~/geovox$ python3.11 -m venv .venv

3) Activate virtual environment (optional)
	<username>:~/geovox$ source .venv/bin/activate

4) Install requirements to the virtual environment.
	(.venv) <username>:~/geovox$ pip install -r requirements.txt

5.a) Install geovox to the virtual environment. This first builds the wheel file into ./dist and then install into .venv from there. See the Makefile.
	(.venv) <username>:~/geovox$ make install

5.b) (Alternate) If the Makefile installation doesn't work, try running the standard installation. This will probably be slower, as it will not compile files in parallel.
	(.venv) <username>:~/geovox$ python -m pip install .

6) Make a directory to store outputs from tests
	(.venv) <username>:~/geovox$ mkdir ./tests/outfiles

7) Run tests from geovox/tests
	(.venv) <username>:~/geovox$ python ./tests/assembly_1.py

	view the results using a .vtk viewer such as Paraview (https://www.paraview.org/). View as a wireframe and extract levelsets from 1<= nvert <= 8 for best results.

8) To clean build files, run:
	(.venv) <username>:~/geovox$ make clean

9) To uninstall, run either of the commands
	(.venv) <username>:~/geovox$ make uninstall

	or 

	(.venv) <username>:~/geovox$ pip uninstall geovox
