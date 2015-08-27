all: clean

clean:
	find . -iname "*.pyc" -delete
	find . -iname "*~" -delete
	rm -rf build/
	rm -rf dist
	rm -rf estimationpy.egg-info/

