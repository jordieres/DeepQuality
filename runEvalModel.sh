#!/bin/bash
#
#
# Check if tensorflow is installed
#

handle_error() {
    echo "An error occurred on line $1 of runEvalModel.sh"
    exit 1
}

a=$(/opt/bitnami/miniconda/bin/pip3 show tensorflow | grep -v WARNING | wc -l )
if [[ $a -ne 0 ]]; then
    /opt/bitnami/miniconda/bin/pip3 install seaborn scikit-learn tensorflow imblearn
fi
#
# Process the feedback and assess the model performance for this coil and for 
# other coils within a period of time.
if [[ $# -ne 1 ]]; then
    echo "Error: wrong number of arguments"
else
    /opt/bitnami/miniconda/bin/python3 /opt/nifi/data/jupyter/upm/scripts/feedback_cnn-model.py -c /opt/nifi/data/jupyter/upm/scripts/keys.yaml -l "$1"
fi
#