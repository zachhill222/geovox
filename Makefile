.PHONY: build install clean uninstall reinstall

build:
	python3 ./setup.py build_ext --inplace -j4 bdist_wheel

install: build
	python3 -m pip install $(shell find ./dist -name "*.whl")

reinstall: uninstall install


clean:
	$(RM) -r build dist *.egg-info
	find ./geovox -name "*.c" -type f -delete
	find ./geovox -name "*.so" -type f -delete
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	pip uninstall geovox -y