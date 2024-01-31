[How to use]

1. download youtube history via google takeout (in json format)

2. put the watch-history.json file in the same folder as everything else

3. open cmd
	cd [path-to-this-folder]
	.\venv\Scripts\activate
	pip install -r requirements.txt
	python class.py

4. it will output the videos classified by categories in the same json format



[Optional]

*if you have a CUDA supported GPU

install cuda from nvidia 	
	https://developer.nvidia.com/cuda-toolkit
run this program
	python classgpu.py



[Settings]

Open the class.py file to change the classifications categories ['Music', 'Education', 'Entertainment', 'Sports', 'Technology', 'Cooking', 'Travel', 'News', 'Gaming']
