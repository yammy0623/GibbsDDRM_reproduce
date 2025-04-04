export CUDA_VISIBLE_DEVICES=1
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 100 --outfile ./exp/result_100.txt
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 10 --outfile ./exp/result_10.txt
python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 5 --outfile ./exp/result_5.txt
