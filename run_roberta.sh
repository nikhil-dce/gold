python main.py --task star --version baseline --learning-rate 1e-5 --model roberta --n-epochs 25 --do-train --do-save
python main.py --task flow --version baseline --learning-rate 3e-5 --model roberta --n-epochs 25 --do-train --do-save
python main.py --task rostd --version baseline --learning-rate 1e-5 --model roberta --n-epochs 25 --do-train --do-save

python main.py --task star --version baseline --model roberta --temperature 1.2 --do-eval --method maxprob
python main.py --task flow --version baseline --model roberta --temperature 1.5 --do-eval --method maxprob
python main.py --task rostd --version baseline --model roberta --temperature 1.8 --do-eval --method maxprob

python main.py --task star --version baseline --model roberta --temperature 1 --do-eval --method entropy
python main.py --task flow --version baseline --model roberta --temperature 1.5 --do-eval --method entropy
python main.py --task rostd --version baseline --model roberta --temperature 2.8 --do-eval --method entropy


python main.py --task star --version baseline --model roberta --do-eval --method bert_embed
python main.py --task flow --version baseline --model roberta --do-eval --method bert_embed
python main.py --task rostd --version baseline --model roberta --do-eval --method bert_embed

python main.py --task star --version baseline --model roberta --do-eval --method mahalanobis
python main.py --task flow --version baseline --model roberta --do-eval --method mahalanobis
python main.py --task rostd --version baseline --model roberta --do-eval --method mahalanobis

python main.py --task star --version baseline --model roberta --do-eval --method dropout
python main.py --task flow --version baseline --model roberta --do-eval --method dropout
python main.py --task rostd --version baseline --model roberta --do-eval --method dropout

python main.py --task star --version baseline --model roberta --do-eval --method odin
python main.py --task flow --version baseline --model roberta --do-eval --method odin
python main.py --task rostd --version baseline --model roberta --do-eval --method odin

python main.py --task star --version baseline --model roberta --batch-size 1 --do-eval --method gradient
python main.py --task flow --version baseline --model roberta --batch-size 1 --do-eval --method gradient
python main.py --task rostd --version baseline --model roberta --batch-size 1 --do-eval --method gradient

python main.py --task star --version baseline --model roberta --do-eval --method nml
python main.py --task flow --version baseline --model roberta --do-eval --method nml
python main.py --task rostd --version baseline --model roberta --do-eval --method nml
