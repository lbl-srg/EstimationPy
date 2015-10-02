all: clean

clean:
	find . -iname "*.pyc" -delete
	find . -iname "*~" -delete
	fing . -iname ".DS_Store" -delete
	rm -rf build/
	rm -rf dist
	rm -rf estimationpy.egg-info/

