#!/bin/bash -ue
if [ -e "user_predict_results.txt" ]; then
    cat user_predict_results.txt
else
    echo "Eroor! Errors in model predictions!"
fi
