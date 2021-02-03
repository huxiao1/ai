from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse



if __name__ == '__main__': #python shortcut_prune.py --percent 0.6
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/sparse-yolov3-full-mAP48.1.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.6, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)


    #****************************************************************
    # add the prune code here

    # 解析模型文件
    def parse_module_defs2(module_defs):
        CBL_idx = []
        Conv_idx = []
        shortcut_idx=dict()
        shortcut_all=set()
        ignore_idx = set()
        for i, module_def in enumerate(module_defs):
            if module_def['type'] == 'convolutional':
                # 如果是卷积层中的BN层则将该层索引添加到CBL_idx
                if module_def['batch_normalize'] == '1': 
                    CBL_idx.append(i)
                else:
                    Conv_idx.append(i)
                if module_defs[i+1]['type'] == 'maxpool':
                    #spp前一个CBL不剪
                    ignore_idx.add(i)

            elif module_def['type'] == 'upsample':
                #上采样层前的卷积层不裁剪
                ignore_idx.add(i - 1)

            elif module_def['type'] == 'shortcut':
                # 根据cfg中的from层获得shortcut的卷积层对应的索引
                identity_idx = (i + int(module_def['from']))
                # 如果shortcut连接的是卷积层则直接添加索引
                if module_defs[identity_idx]['type'] == 'convolutional':

                    #ignore_idx.add(identity_idx)
                    shortcut_idx[i-1]=identity_idx
                    shortcut_all.add(identity_idx)
                # 如果shortcut连接的是shortcut层，则添加前一层卷积层的索引
                elif module_defs[identity_idx]['type'] == 'shortcut':

                    #ignore_idx.add(identity_idx - 1)
                    shortcut_idx[i-1]=identity_idx-1
                    shortcut_all.add(identity_idx-1)
                shortcut_all.add(i-1)
            # 得到需要剪枝的BN层的索引
        prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
        return CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all

    eval_model = lambda model:test(model=model, cfg=opt.cfg, data=opt.data, batch_size=10, img_size=img_size)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])
    origin_nparameters = obtain_num_parameters(model)

    CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all = parse_module_defs2(model.module_defs)
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    sorted_bn = torch.sort(bn_weights)[0]

    def prune_and_eval(model, sorted_bn, percent=.0):
        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn)*percent)
        threl = sorted_bn[thre_index]

        print(f'Channels with Gamma value less than {threl  10f} are pruned!')

        total, remain_num = 0,0
        idx_new = dict()

        for idx in prune_idx:
            if idx not in shortcut_idx:
                bn_module = model_copy.module_list[idx][1]
                mask = obtain_bn_mask(bn_module, threl)
                idx_new[idx] = mask
                bn_module.weight.data.mul_(mask)
            else:
                bn_module = model_copy.module_list[idx][1]
                mask = idx_new[shortcut_idx[idx]]
                idx_new[idx] = mask
                bn_module.weight.data.mul_(mask)
            remain_num += int(mask.sum())
            total += mask.shape[0]
        print(f'Num. of channels has been reduced from {total} to {remain_num}')

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]
        return threl
    percent = opt.percent
    threshold = prune_and_eval(model, sorted_bn, percent)

    #****************************************************************
    #虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型

    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):
        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        idx_new=dict()
        #CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:
                if idx not in shortcut_idx:
                    mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                    idx_new[idx]=mask
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                else:
                    mask=idx_new[shortcut_idx[idx]]
                    idx_new[idx]=mask
                    remain= int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain                
                if remain == 0:
                    print("Channels would be all pruned!")
                    # raise Exception
                    max_value = bn_module.weight.data.abs().max()
                    mask = obtain_bn_mask(bn_module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                        f'remaining channel: {remain:>4d}')
            else:
                mask = np.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]
            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())
        #因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}/{total}\tPrune ratio: {prune_ratio:.3f}')
        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)
    #CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    #获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        torch.cuda.synchronize()
        avg_infer_time = (time.time() - start) / repeat
        return avg_infer_time, output

    print('testing Inference time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing final model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)

    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        #["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["mAP", f'{0.481}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)

    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{percent}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f'Config file has been saved: {pruned_cfg_file}')

    compact_model_name = opt.weights.replace('/', f'/prune_{percent}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, path=compact_model_name)
    print(f'Compact model has been saved: {compact_model_name}')

