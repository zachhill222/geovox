.PHONY: build dist redist install install-from-source clean uninstall

build:
	python3 ./setup.py build_ext --inplace

dist:
	python3 ./setup.py sdist

# redist: clean dist

install:
	python3 -m pip install .

# install-from-source: dist #Doesn't seem to work at the moment. ValueError: 'geovox/utilities/vector3.pyx' doesn't match any files
# 	python3 -m pip install dist/geovox-0.1.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	find ./geovox -name "*.c" -type f -delete
	find ./geovox -name "*.so" -type f -delete
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	pip uninstall geovox