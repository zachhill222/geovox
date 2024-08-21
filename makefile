clean:
	rm -rf build
	rm -rf __pycache__ 
	rm -rf geovox/__pycache__
	find ./geovox -name "*.c" -type f -delete
	find ./geovox -name "*.so" -type f -delete