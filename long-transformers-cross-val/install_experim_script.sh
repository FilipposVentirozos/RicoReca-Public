#!/bin/bash

echo "1. Installing Transformers"
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout cbecf121cdeff6fb7193471cf759f9d734e37ea9
python -m venv venv
source venv/bin/activate
which pip
pip install -e .
pip install torch # This may differ according to your CUDA version, view https://pytorch.org/get-started/locally/
pip install evaluate 
pip install rouge-score

cd ..


echo "2. Run Training"

python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold0_pegasus.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold1_pegasus.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold2_pegasus.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold3_pegasus.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold4_pegasus.json

python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold0_t5.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold1_t5.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold2_t5.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold3_t5.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold4_t5.json




echo "3. Run Prediction"

python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold0_t5_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold1_t5_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold2_t5_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold3_t5_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold4_t5_pred.json

python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold0_pegasus_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold1_pegasus_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold2_pegasus_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold3_pegasus_pred.json
python transformers/examples/pytorch/summarization/run_summarization.py hyper-params/fold4_pegasus_pred.json

shutdown
