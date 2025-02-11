docker run -it --rm \
      -v ./bert-code-new:/bert-code-new \
	    -v ./chinese-roberta-wwm-ext-l12-h768-a12:/chinese-roberta-wwm-ext-l12-h768-a12 \
      -v ./test-for-inference:/test-for-inference \
	    -v ./output_predict:/output_predict \
	    chinese-sentiment-analysis:latest \
	    python ./bert-code-new/run_classifier.py \
        --task_name=mytask \
        --do_predict=true \
        --data_dir=test-for-inference \
        --vocab_file=chinese-roberta-wwm-ext-l12-h768-a12/vocab.txt \
        --bert_config_file=chinese-roberta-wwm-ext-l12-h768-a12/bert_config.json \
        --init_checkpoint=chinese-roberta-wwm-ext-l12-h768-a12 \
        --max_seq_length=512 \
        --output_dir=output_predict
