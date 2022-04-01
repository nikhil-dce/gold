python main.py --task star --version baseline --learning-rate 1e-5 --model bert --n-epochs 25 --do-train --do-save --mixup 1
python main.py --task flow --version baseline --learning-rate 3e-5 --model bert --n-epochs 25 --do-train --do-save --mixup 1
python main.py --task rostd --version baseline --learning-rate 1e-5 --model bert --n-epochs 25 --do-train --do-save --mixup 1

python main.py --task star --version baseline --temperature 1.2 --do-eval --method maxprob
python main.py --task flow --version baseline --temperature 1.5 --do-eval --method maxprob
python main.py --task rostd --version baseline --temperature 1.8 --do-eval --method maxprob

python main.py --task star --version baseline --temperature 1 --do-eval --method entropy
python main.py --task flow --version baseline --temperature 1.5 --do-eval --method entropy
python main.py --task rostd --version baseline --temperature 2.8 --do-eval --method entropy


python main.py --task star --version baseline --do-eval --method bert_embed
python main.py --task flow --version baseline --do-eval --method bert_embed
python main.py --task rostd --version baseline --do-eval --method bert_embed

python main.py --task star --version baseline --do-eval --method mahalanobis
python main.py --task flow --version baseline --do-eval --method mahalanobis
python main.py --task rostd --version baseline --do-eval --method mahalanobis

python main.py --task star --version baseline --do-eval --method dropout
python main.py --task flow --version baseline --do-eval --method dropout
python main.py --task rostd --version baseline --do-eval --method dropout

python main.py --task star --version baseline --do-eval --method odin
python main.py --task flow --version baseline --do-eval --method odin
python main.py --task rostd --version baseline --do-eval --method odin

python main.py --task star --version baseline --batch-size 1 --do-eval --method gradient
python main.py --task flow --version baseline --batch-size 1 --do-eval --method gradient
python main.py --task rostd --version baseline --batch-size 1 --do-eval --method gradient

python main.py --task star --version baseline --do-eval --method nml
python main.py --task flow --version baseline --do-eval --method nml
python main.py --task rostd --version baseline --do-eval --method nml
