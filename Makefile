.PHONY: build install clean uninstall

build:
	python3 ./setup.py build_ext --inplace -j4

install:
	python3 -m pip install .

clean:
	$(RM) -r build *.egg-info
	find ./geovox -name "*.c" -type f -delete
	find ./geovox -name "*.so" -type f -delete
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	pip uninstall geovox