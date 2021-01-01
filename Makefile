# ----------------------------------
#          GCP
# ----------------------------------
JOB_NAME=nyc_taxifare_predictor_$(shell date +'%Y%m%d_%H')
BUCKET_NAME=nyc_taxifare_predictor
BUCKET_TRAINING_FOLDER=trainings
PACKAGE_NAME=NYCtaxifarePredictor
FILENAME=trainer
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.3
REGION=europe-west1

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


train_on_gcp:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
					--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
					--package-path ${PACKAGE_NAME} \
					--module-name ${PACKAGE_NAME}.${FILENAME}	\
					--python-version=${PYTHON_VERSION} \
					--runtime-version=${RUNTIME_VERSION} \
					--region ${REGION} \
					--stream-logs

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

test:
	@coverage run -m pytest tests/*.py
	#@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr NYCtaxifarePredictor-*.dist-info
	@rm -fr NYCtaxifarePredictor.egg-info

install:
	@pip install . -U

all: clean install test


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2
