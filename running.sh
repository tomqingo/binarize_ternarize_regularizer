echo "running ternary regularizer scripts"


echo "running l2 regularizer"
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save l2_5e-6 --weight_decay 5e-6 --regularize_type l2
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save l2_5e-8 --weight_decay 5e-8 --regularize_type l2
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save l2_5e-5 --weight_decay 5e-5 --regularize_type l2

echo "running poly regularizer"
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save poly_5e-6 --weight_decay 5e-6 --regularize_type poly
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save poly_5e-8 --weight_decay 5e-8 --regularize_type poly
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save poly_5e-5 --weight_decay 5e-5 --regularize_type poly

echo "running l2 regularizer"
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save sine_5e-6 --weight_decay 5e-6 --regularize_type sine
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save sine_5e-8 --weight_decay 5e-8 --regularize_type sine
python main_ternary.py --model vgg_cifar10_ternary --epochs 200 --save sine_5e-5 --weight_decay 5e-5 --regularize_type sine


