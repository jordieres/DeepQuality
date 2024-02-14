#!/bin/bash
#
#
# Check if tensorflow is installed
#
handle_error() {
    echo "An error occurred on line $1 of runEvalCoil.sh"
    exit 1
}
a=$(/opt/bitnami/miniconda/bin/pip3 show tensorflow | grep -v WARNING | wc -l )
if [[ $a -ne 0 ]]; then
    /opt/bitnami/miniconda/bin/pip3 install seaborn scikit-learn tensorflow imblearn
fi
#
# Create the scaled information realted to tiles
if [[ $# -ne 1 ]]; then
    echo "Error: wrong number of arguments"
else
    /opt/bitnami/miniconda/bin/python3 /opt/nifi/data/jupyter/upm/scripts/prep_features.py -v -v -c /opt/nifi/data/jupyter/upm/scripts/keys.yaml -p /opt/nifi/data/jupyter/upm/pkls/ -l "$1"
fi
#
# Check that the normalized data for the coil does exist
datf=/opt/nifi/data/jupyter/upm/pkls/norm_data_"$1".pkl
#
# Run the models to assess the Coil.
if [[ -f $datf ]]; then
    /opt/bitnami/miniconda/bin/python3 /opt/nifi/data/jupyter/upm/scripts/test_cnn-model.py -v -v -c /opt/nifi/data/jupyter/upm/scripts/keys.yaml -p /opt/nifi/data/jupyter/upm/pkls/norm_data_"$1".pkl
else
    echo "Error: Coil $1 didn't create the fine norm_data_$1.pkl successfully! "
fi
