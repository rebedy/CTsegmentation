import os
import sys
from os.path import join
from glob import glob
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.jit import script, trace       # hybrid frontend decorator and tracing jit

from args import parse_args
from data_io.custom_dataset import PatientDataset, CTSeg_BatchSliceDataset, CTSeg_ValDataset
from models import get_model
from loss import get_loss
from misc.utils import makedirs_ifnot

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# ------------------------------------------------------- #


print("\n")
print("┌───────────────────────┐")
print("│ CT Segmentaiton Start │")
print("└───────────────────────┘\n")
start = datetime.datetime.now()
print("     @ Starts at %s" % str(start))

# ### ! Args Setting ! ###
args = parse_args()
TODAY = str(datetime.date.today().strftime('%y%m%d'))
NOW = str(datetime.datetime.now().strftime('_%Hh%Mm'))
SAVE_CHECKPOINT = True

CWD = os.getcwd()
LOG_DIR = CWD + "/_Logs"
SOURCE = args.NAS + "/Data/CT/" + args.DTYPE + "/" + args.args.source_company + "/" + args.args.source_company + "_exported/"
CONVERTED_DIR = args.NAS + "/Data/CT/" + args.DTYPE + "/" + args.args.source_company + "/vti/"
seg_dir = join(LOG_DIR, args.project_tag + 'logs_' + TODAY + NOW + '_' + args.source_company + '_' + args.DTYPE + '_' + args.load_mode)
if args.transfer:
    SEGMENTED_DIR = makedirs_ifnot(seg_dir + '_TL')
else:
    SEGMENTED_DIR = makedirs_ifnot(seg_dir)
INFERENCE_DIR = makedirs_ifnot(join(SEGMENTED_DIR, 'infer/'))

log_name = join(SEGMENTED_DIR, "log_" + TODAY + NOW + '-' + args.source_company + '_' + args.DTYPE + ".log")
logging.basicConfig(filename=log_name, level=logging.INFO, format='%(levelname)s : %(message)s')

# ! --------------------------------------------------------------------------- !


# ### ! NETWORK and HYPER-PARAM! ###
# ### * Device Setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f'>>> Using device: {device}')

# ### * Neural Network !!!
net = get_model(args.net, args)
net.to(device=device)
if args.transfer:
    net.load_state_dict(torch.load(join(args.predicted_log_dir, args.pretrained_model), map_location=device))
    logging.info(f'>>> PRETRAINED_MODEL: {args.pretrained_model}')
else:
    pass

# ### Optimizer
params = net.parameters()
if args.optimizer == 'adam':
    opti = optim.Adam(params, lr=args.lr)  # weight_decay=LD)
elif args.optimizer == 'sgd':
    opti = optim.SGD(params, lr=args.lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
elif args.optimizer == 'rmsprop':
    opti = optim.Rprop(params, lr=args.lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

# ### Loss Function
criterion = get_loss(args.loss, args.out_dim, thresh=0.7, weight=None, ignore_index=255)

scheduler = lr_scheduler.ReduceLROnPlateau(opti, 'min' if args.args.out_dim > 1 else 'max', patience=2)

# ### TODO | DY : * SummaryWriter; to make logs for tensorboard(visualization)
tb_summary_writer = SummaryWriter(log_dir=SEGMENTED_DIR, comment=f'LR_{args.lr}_BS_{args.args.pati_batch}_EP_{args.args.epoch}')

# ! --------------------------------------------------------------------------- !


# ### ! DATASET PATH and LOADER ! ###
dataset_path_list = glob(CONVERTED_DIR + "/*")[:]

pati_indices = list(range(len(dataset_path_list)))
split_idx = int(np.floor(args.val_split * len(dataset_path_list)))
if args.pati_shuffle:
    np.random.seed(args.seed)
    np.random.shuffle(pati_indices)
train_idx, val_idx = pati_indices[split_idx:], pati_indices[:split_idx]
n_train, n_val = len(train_idx), len(val_idx)  # 144, 16
pati_dataset = PatientDataset(dataset_path_list)
train_paths = DataLoader(pati_dataset, batch_size=args.pati_batch, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(train_idx))
valid_paths = DataLoader(pati_dataset, batch_size=args.pati_batch, shuffle=False, num_workers=0, sampler=SubsetRandomSampler(val_idx))
print('     >>> CONVERTED_DIR : ', CONVERTED_DIR)
print('     >>> SEGMENTED_DIR : ', SEGMENTED_DIR)
print('     >>> len(dataset_path_list) : ', len(dataset_path_list))
print('     >>> len(train_idx) : ', n_train)
print('     >>> len(val_idx) : ', n_val)

logging.info(f'''>>> Start training!
                args.source_company: {args.source_company}
                DTYPE:          {args.DTYPE}
                args.load_mode:      {args.load_mode}
                Epochs:         {args.args.epoch}
                Batch size:     {args.args.slice_batch}
                Learning rate:  {args.lr}
                Optimizer:      {args.optimizer}
                Criterion:      {args.loss}
                Scheduler:      {scheduler}
                Train size:     {n_train}
                Valid size:     {n_val}
                ''')

logging.info(">>> Validation Patients")
val_patients = [next(iter(valid_paths))[0].split('\\')[-1] for _ in range(len(valid_paths))]
by_ = 10
for i in range(int(len(val_patients) / by_)):
    logging.info(f"{val_patients[i*by_:(i+1)*by_]}")
# ! --------------------------------------------------------------------------- !


# ! ------------ !
# !     MAIN     !
# ! ------------ !
global_step = 0
ep_loss = 0.0
train_losses, valid_losses = [], []
best_acc = 0.0

# ### ! args.epoch Start ! ###
for ep in range(args.args.epoch):
    start_epoch = datetime.datetime.now()
    print("\n\n=============== [args.epoch %d] " % (ep + 1), str(start_epoch), ' ===============\n')
    logging.info(f'=============== [args.epoch {str(ep+1)}] {str(start_epoch)} ===============')

    # ### ! Training Loop ! ###
    tr_pati_loss = 0
    for p, patient_dir in enumerate(train_paths):
        tr_pati_num = patient_dir[0].split('\\')[-1]
        # start_training_pati = datetime.datetime.now()

        # ### ! * Training Dataset Load !
        CTSeg_slices = CTSeg_BatchSliceDataset(patient_dir[0])
        ct_depth = len(CTSeg_slices)  # == len(train_loader)
        # ### TODO | DY : You cannot true the shuffle arg. here because of indexing.(See custom_dataset.py)
        train_loader = DataLoader(CTSeg_slices, batch_size=args.slice_batch, shuffle=False, num_workers=0)
        # , pin_memory=False, drop_last=False, timeout=0)
        # ### https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/5

        net.train()

        tr_loss, train_iou = 0.0, 0.0
        for idx, batch in enumerate(train_loader):
            input_tensor = batch['image'].to(device)
            gt_tensor = batch['mask'].to(device)

            # ### ! * Actual Training
            opti.zero_grad()   # ### zero the gradient buffers; 변화도 버퍼를 0으로 설정
            pred = net(input_tensor)  # Output
            loss = criterion(pred, gt_tensor)
            loss.backward()   # ### Back Propagation
            opti.step()    # ### Does the update!!!

            tr_loss += loss.item()

            tb_summary_writer.add_scalar('9_ori/Loss/train', loss.item(), global_step)
            tb_summary_writer.add_scalars('0_Global_Step/Compare_TL_LR', {'Train_Loss': loss.item(), 'LR': opti.param_groups[0]['lr']}, global_step)
            tb_summary_writer.add_scalar('0_Global_Step/Learning_Rate', opti.param_groups[0]['lr'], global_step)
            tb_summary_writer.add_scalar('0_Global_Step/Train_Loss', loss.item(), global_step)

            # ### in your training loop:
            global_step += 1
            # if global_step % (n_train*100 // (10 * input_tensor.shape[1])) == 0:
            #     for tag, value in net.named_parameters():
            #         tag = tag.replace('.', '/')
            #         tb_summary_writer.add_histogram('9_ori/weights/' + tag, value.data.cpu().numpy(), global_step)
            #         tb_summary_writer.add_histogram('9_ori/grads/' + tag, value.grad.data.cpu().numpy(), global_step)

        train_loss = tr_loss / ct_depth  # Average loss of each patient's batch
        print("▻  {}th Training Iteration | Patient Training Loss(mean) {} | Patient {}"
              .format(str(p + 1), train_loss, tr_pati_num))
        # logging.info(f"▶ {str(p+1)}th Training Iteration | Patient Training Loss(mean) {train_loss} | Patient {tr_pati_num}")
        tr_pati_loss += train_loss  # Sum of each patient's average batch loss

    torch.save(net.state_dict(), SEGMENTED_DIR + f'/checkpoints_ep{ep + 1}.pth')

    # CKPT_DIR_PER_EP = makedirs_ifnot(join(SEGMENTED_DIR, f'checkpoints_ep{ep + 1}'))
    #         # f'ckpt_{TODAY}{NOW}_{args.source_company}_{DTYPE}_{args.load_mode}_BS{args.slice_batch}_EP{ep+1}'))
    # ### Save CPU Model
    # net.to(device="cpu")
    # dummy_input = dummy_input.to("cpu")
    # module_cpu = torch.jit.trace(net.forward, dummy_input)
    # module_cpu.save(join(CKPT_DIR_PER_EP, "sample_model_cpu.pt" ))
    # ### Save CUDA Model
    # net.to(device="cuda")
    # dummy_input = dummy_input.to("cuda")
    # module_cuda = torch.jit.trace(net.forward, dummy_input)
    # module_cuda.save(join(CKPT_DIR_PER_EP,"sample_model_cuda.pt"))

    train_itr_done = datetime.datetime.now() - start_epoch
    print("\n   >>> %d th args.epoch training DONE! %s\n" % (ep + 1, str(train_itr_done)))
    logging.info('   >>> %d th args.epoch training DONE! %s' % (ep + 1, str(train_itr_done)))
    print("-----------------------------------------------------------------------")

    # ### ! Validation Loop ! ###
    start_validation = datetime.datetime.now()
    val_pati_loss, val_patients = 0.0, []
    global_eval_step = 0

    for v, val_patient in enumerate(valid_paths):
        val_num = val_patient[0].split('\\')[-1]
        if ep + 1 == args.epoch:
            val_patients.append(val_num)

        # ### * Validation Dataset Load !
        val_CTSeg_slices = CTSeg_ValDataset(val_patient[0])
        val_ct_depth = len(val_CTSeg_slices)  # == len(valid_loader) # the number of batch
        valid_loader = DataLoader(val_CTSeg_slices, batch_size=args.slice_batch, shuffle=False, num_workers=0)

        """Evaluation without the densecrf with the dice coefficient"""
        net.eval()

        prediction = np.zeros((val_ct_depth, args.out_dim))
        v_loss, v_acc, v_iou = 0.0, 0.0, 0.0  # ep_loss
        for v_idx, val_batch in enumerate(valid_loader):
            # ? if v_idx < 2 or v_idx > val_ct_depth-3 : continue
            val_imgs, val_true_masks, val_spacing = val_batch['image'], val_batch['mask'], val_batch['image_sp']
            # print(val_imgs.shape, val_true_masks.shape) ### [1, 1, 512, 512] & [1, 512, 512]
            val_imgs = val_imgs.to(device)
            val_true_masks = val_true_masks.to(device)

            # ### ! * Actual Validation
            with torch.no_grad():  # 자동미분 off
                val_pred = net(val_imgs)
                # val_loss = criterion(val_pred, val_true_masks)
                val_loss = F.cross_entropy(val_pred, val_true_masks)  # For valid
                v_loss += val_loss.item()

                tb_summary_writer.add_scalar('9_ori/Loss/ valid', val_loss.item(), global_eval_step)
                tb_summary_writer.add_scalar('9_ori/learning_rate', opti.param_groups[0]['lr'], global_eval_step)
                # https://www.learnopencv.com/experiment-logging-with-tensorboard-and-wandb/
                # For logging
                # https://www.kaggle.com/mlagunas/naive-unet-with-pytorch-tensorboard-logging

            tb_summary_writer.add_scalar('0_Global_Step/Validation_Loss', val_loss.item(), global_eval_step)
            global_eval_step += 1

            scheduler.step(val_loss)
        # ! --------------------------------------------------------------------------- !

        # ### *  Mean validation score for tot. batch of one valid_patient.
        valid_loss = v_loss / val_ct_depth  # val loss ->

        val_pati_loss += valid_loss
        # valid_loss = np.round(valid_loss, 2)
        print("▻  {}th Validation | Loss(mean) {} | Patient {}"
              .format(str(v + 1), valid_loss, val_num))
        # logging.info(f"▶ {str(v+1)}th Validation | Loss(mean) {valid_loss} | Patient {val_num}")

    valid_itr_done = datetime.datetime.now() - start_validation
    print("\n   >>> %d th args.epoch validation DONE! %s\n" % (ep + 1, str(valid_itr_done)))
    logging.info('   >>> %d th args.epoch validation DONE! %s' % (ep + 1, str(valid_itr_done)))
    print("-----------------------------------------------------------------------")
    # ! --------------------------------------------------------------------------- !

    net.train()
    running_loss = 0

    tb_summary_writer.close()

    # ### * After each args.epoch, print the logits important.
    mean_train_cross_entropy = tr_pati_loss / n_train  # len(train_paths)
    mean_val_cross_entropy = val_pati_loss / n_val
    print('\n>>> *mean_train_cross_entropy : ', mean_train_cross_entropy)
    print('>>> *mean_val_cross_entropy : ', mean_val_cross_entropy)
    tb_summary_writer.add_scalar('1_Each_Epoch/Train_Loss', mean_train_cross_entropy, ep + 1)
    tb_summary_writer.add_scalar('1_Each_Epoch/Validation_Loss', mean_val_cross_entropy, ep + 1)
    train_losses.append(mean_train_cross_entropy)  # 전체 환자의 평균 loss
    valid_losses.append(mean_val_cross_entropy)

    epoch_done = datetime.datetime.now() - start_epoch
    print('\n>>> %d th args.epoch Done! It took for  %s' % (ep + 1, str(epoch_done)))
    print("=======================================================================")
    # ! --------------------------------------------------------------------------- !

    logging.info(f'''>>> *mean_train_cross_entropy : {mean_train_cross_entropy}
                 >>> *mean_val_cross_entropy : {mean_val_cross_entropy}
                 >>> {str(ep+1)} th args.epoch Done!!! Process Time : {str(epoch_done)}
                 =======================================================================\n
                 ''')

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig(log_name[:-4] + '_plot.png', dpi=300)

print('\n\n>>> Avg. Total Train Loss : ', sum(train_losses) / args.epoch)
print('>>> Avg. Total Valid Loss : ', sum(valid_losses) / args.epoch)
logging.info(f'''\n\n
             >>> Avg. Total Train Loss : {sum(train_losses)/args.epoch}
             >>> Avg. Total Valid Loss : {sum(valid_losses)/args.epoch}

             >>> Whole process time : {str(datetime.datetime.now()-start)}
             >>> Now : {str(datetime.datetime.now())}
             =======================================================================\n
             ''')


print('\n\n>>> Whole process took for  %s' % (str(datetime.datetime.now() - start)))
print(datetime.datetime.now())
print("=======================================================================\n")
