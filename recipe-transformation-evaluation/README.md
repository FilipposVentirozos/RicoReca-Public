# Evaluations

## Running the Evaluations
You can check the inter-anntotator agreements, between the two annotators

`evaluation.py --eval-type inter`

The 5-Fold Cross Validation 

`evaluation.py --eval-type cross`

Also, should you want to do the inference with PEGASUS-X and LongT5 you can run 
`evaluation.py --gen-cross`

Inside the code you can also change the random seed for the 5-fold Cross Validation and you can visualise the graphs produced by PEGASUS-X and LongT5.
Should you need assistance, please open an issue.