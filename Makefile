train_CAG:
	CUDA_VISIBLE_DEVICES='4' \
	python src/main.py \
		--dataset Decagon \
		--phase train \
		--config multi \
		--model CAGPool \
		--init xavier \
		--gcn_type gcn \
		--num_layers 2

train_SAG:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--dataset Decagon \
		--phase train \
		--config multi \
		--model SAGPool \
		--init xavier \
		--gcn_type gcn \
		--num_layers 2

train_TopK:
	CUDA_VISIBLE_DEVICES='4' \
	python src/main.py \
		--dataset Decagon \
		--phase train \
		--config multi \
		--model TopKPool \
		--init xavier \
		--gcn_type gcn \
		--num_layers 2

test_CAG:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--dataset Decagon \
		--phase test \
		--config multi \
		--model CAGPool \
		--init xavier \
		--gcn_type gcn \
		--num_layers 2 
