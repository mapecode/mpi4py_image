RUN := mpirun

default:
    # Example: make default nodes=4 img=../photo.jpg filter=emboss
	$(RUN) -np $(nodes) python kernel_filters.py $(img) $(filter)

test:
	# Example: make test img=../photo.jpg filter=emboss
	$(RUN) -np 1 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 2 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 3 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 4 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 5 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 6 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 7 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 8 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 9 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 10 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 20 python kernel_filters.py $(img) $(filter)
	$(RUN) -np 50 python kernel_filters.py $(img) $(filter)

all_filters:
	# Example: make all_filters nodes=4 img=../photo.jpg
	$(RUN) -np $(nodes) python kernel_filters.py $(img) blur
	$(RUN) -np $(nodes) python kernel_filters.py $(img) box_blur
	$(RUN) -np $(nodes) python kernel_filters.py $(img) gaussian_blur3x3
	$(RUN) -np $(nodes) python kernel_filters.py $(img) gaussian_blur5x5
	$(RUN) -np $(nodes) python kernel_filters.py $(img) bottom_sobel
	$(RUN) -np $(nodes) python kernel_filters.py $(img) left_sobel
	$(RUN) -np $(nodes) python kernel_filters.py $(img) right_sobel
	$(RUN) -np $(nodes) python kernel_filters.py $(img) top_sobel
	$(RUN) -np $(nodes) python kernel_filters.py $(img) emboss
	$(RUN) -np $(nodes) python kernel_filters.py $(img) identity
	$(RUN) -np $(nodes) python kernel_filters.py $(img) outline
	$(RUN) -np $(nodes) python kernel_filters.py $(img) sharpen
	$(RUN) -np $(nodes) python kernel_filters.py $(img) enhance

clean:
	rm -rf *blur* *sobel* emboss* identity* outline* sharpen* enhance*
	rm -rf __pycache__ *pyc

install:
	sudo apt-get install libopenmpi-dev python-dev
	pip install -r requirements.txt