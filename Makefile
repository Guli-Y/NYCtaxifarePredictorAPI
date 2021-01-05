# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

clean:
	@rm -f */version.txt
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr NYCtaxifarePredictor-*.dist-info
	@rm -fr NYCtaxifarePredictor.egg-info

install:
	@pip install -e .

all: clean install test

uninstall:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

# ----------------------------------
#          API
# ----------------------------------
deploy_heroku:
	-@git push heroku master
	-@heroku ps:scale web=1
	-@heroku config:set GOOGLE_APPLICATION_CREDENTIALS='$(< /Users/Guli/code/Guli-Y/keys/gcp_keys/wagon-project-guli-8ba06c8ff833.json)'

