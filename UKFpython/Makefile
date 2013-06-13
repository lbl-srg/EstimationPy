all: clean add

clean:
	find . -iname "*.pyc" -delete
	find . -iname "*~" -delete

add:
	find . -iname "*.py" -exec git add '{}' \;

