export CUDA_VISIBLE_DEVICES=1
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 100 --outfile ./exp/result_100.txt
# python main.py --ni --config ffhq_256_deblur.yml --exp /tmp2/ICML2025/ffhq --doc ffhq --deg deblur_arbitral -i deblur_ffhq --steps 10 --outfile ./exp/result_10.txt
python main.py --ni --config gopro_blur_gamma_256_deblur.yml --exp /tmp2/ICML2025/GoPro --doc gopro --deg deblur_arbitral -i deblur_gopro --steps 5 --outfile ./exp/gorpo_result_5_resize.txt
# python main.py --ni --config gopro_blur_gamma_256_deblur.yml --exp /tmp2/ICML2025/GoPro --doc gopro --deg deblur_arbitral -i deblur_gopro --steps 100 --outfile ./exp/gorpo_result_100.txt
