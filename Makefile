all:
	$(MAKE) -C src all

clean:
	rm -rf src/__pycache__
	rm -rf src/*.log
	rm -rf src/*_results
