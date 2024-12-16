#!/bin/bash -ue
if [ -e "valid_evaluation.txt" ]; then
    cat valid_evaluation.txt
else
    echo "Eroor! Errors in model predictions!"
fi
