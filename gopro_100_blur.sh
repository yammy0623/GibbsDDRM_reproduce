export CUDA_VISIBLE_DEVICES=0d
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 100 --outfile ./exp/result_100.txt
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 10 --outfile ./exp/result_10.txt
python main.py --ni --config gopro_blur_gamma_256_deblur.yml --datatype "blur" --gopro_H --exp /tmp2/ICML2025/GoPro --doc gopro --deg deblur_arbitral -i deblur_gopro --steps 100 --outfile ./exp/gorpo_blur_result_100_original.txt 
# python main.py --ni --config gopro_blur_gamma_256_deblur.yml --exp /tmp2/ICML2025/GoPro --doc gopro --deg deblur_arbitral -i deblur_gopro --steps 100 --outfile ./exp/gorpo_result_100.txt
