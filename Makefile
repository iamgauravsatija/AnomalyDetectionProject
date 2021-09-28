install_dev_requirements:
	pip install -r requirements_dev.txt

install_requirements:
	pip install -r requirements.txt

formatting:
	safety check
	isort .
	black .
	flake8 .

generate_documentation:
	pdoc --html --output-dir docs --force . 

open_documentation:
	open docs/AnomalyDetectionProject/index.html


run_shortest_dist_path:
	python3 main.py

docker_build:
	docker build . -t iamgauravsatija/anomaly_detection_project 

docker_run:
	docker run --name anomaly_detection_project --env="DISPLAY" --mount  type=bind,source="$$(pwd)/",target=/anomaly_detection_project  iamgauravsatija/anomaly_detection_project

docker_push:
	docker push iamgauravsatija/anomaly_detection_project