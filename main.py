# Multi-source Feature Alignment and Label Rectification (MFA-LR)
# Author:   Magdiel Jiménez-Guarneros
# Article:  Jiménez-Guarneros Magdiel, Fuentes-Pineda Gibran (2023). Learning a Robust Unified Domain Adaptation
#           Framework for Cross-subject EEG-based Emotion Recognition. Biomedical Signal Processing and Control.
# Python 3.6, Pytorch 1.9.0+cu102

import argparse
from solvers import MFA_LR, RSDA
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import os
import numpy as np
import random
from utils import seed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._data")

def main(args):
    args.log_file.write('\n\n###########  initialization ############')
    acc = []
    for i in range(15):
        args.target = 15 - i # i+ 1
        print("当前的目标域是",args.target)
    # [Multi-source Feature Alignment (MFA)]
        X, Y, acc_pre, f1_pre, auc_pre, mat_pre, model, log_loss = MFA_LR(args)
    
        args.test_interval = 50
        # [Label rectification: RSDA]
        for stage in range(args.stages):
            
            print('\n\n########### stage : {:d}th ##############\n\n'.format(stage+1))
            args.log_file.write('\n\n########### stage : {:d}th    ##############'.format(stage+1))
            
            # assign pseudo-labels
            # samples, weighted_pseu_label, weights = make_weighted_pseudo_list(X, Y, args, model)
    
            # [single-source domain adaptation]
            acc_final, f1_final, auc_final, mat, model = RSDA(X, Y, model, args)
            acc.append(acc_final)
            print('这个人的准确率：', acc)
    print('15个人的准确率：', acc)
    print('平均准确率：', np.mean(acc))
    list_metrics_classification = []
    list_metrics_classification.append([acc_pre, f1_pre, auc_pre, acc_final, f1_final, auc_final, args.session, args.target, args.seed])
    list_metrics_clsf = np.array(list_metrics_classification)

    # [Save classification results]

    save_file_metrics = "outputs/RSDA-" + args.dataset +"-session-" + str(args.session) + "-metrics-classification.csv"
    directory = os.path.dirname(save_file_metrics)  # 获取目录路径
    if not os.path.exists(directory):
        os.makedirs(directory)  # 创建目录
    f = open(save_file_metrics, 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--dataset',type=str,default='seed')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1, help="Iteration repetitions")
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--mixed_sessions', type=str, default='per_session', help="[per_session | mixed]")
    parser.add_argument('--lr_a', type=float, default=0.1, help="learning rate 1")
    parser.add_argument('--lr_b', type=float, default=0.1, help="learning rate 2")
    parser.add_argument('--radius', type=float, default=10, help="radius")
    parser.add_argument('--num_class',type=int,default=3,help='the number of classes')
    parser.add_argument('--stages', type=int, default=1, help='the number of alternative iteration stages')
    parser.add_argument('--max_iter1',type=int,default=100)
    parser.add_argument('--max_iter2', type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=36)
    parser.add_argument('--seed', type=int, default=123, help="random seed number ")
    parser.add_argument('--hidden_size', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=1, help="random seed number ")
    parser.add_argument('--file_path', type=str, default='/home/Descargas/', help="Path from the current dataset")
    parser.add_argument('--log_file')
    #####
    parser.add_argument('--ila_switch_iter', type=int, default=1, help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--mu', type=int, default=80, help="these many target samples are used finally, eg. 2/3 of batch")  # mu in number
    parser.add_argument('--k', type=int, default=3, help="k")
    parser.add_argument('--msc_coeff', type=float, default=1.0, help="coeff for similarity loss")
    #####
    args = parser.parse_args()

    # 固定随机种子
    seed(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    # 创建存储结果文件
    if not os.path.exists('snapshot'):
        os.mkdir('snapshot')
    if not os.path.exists('snapshot/{}'.format(args.output_dir)):
        os.mkdir('snapshot/{}'.format(args.output_dir))
    log_file = open('snapshot/{}/log.txt'.format(args.output_dir),'w')
    log_file.write('dataset:{}\tsource:{}\ttarget:{}\n\n'
                   ''.format(args.dataset,args.source, str(args.target)))
    args.log_file = log_file

    # 导入数据集地址
    if args.dataset == "seed":
        args.file_path = "/home/lyc/research/research_5/DA/ExtractedFeatures/"
    elif args.dataset == "seed-iv":
        args.file_path = "/home/magdiel/Data/SEED-IV/eeg/"
    else:
        print("This dataset does not exist.")
        exit(-1)


    main(args)




