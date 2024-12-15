# configpath=config/jl_4326.yaml

configpath=config/hebei.yaml

python 00-buildpyarmia.py $configpath
python 01-bulidbox.py $configpath
python 02-genfeature.py $configpath