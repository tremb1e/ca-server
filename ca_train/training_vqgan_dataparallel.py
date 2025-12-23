import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from data_loaders import CIFAR10DataLoader_pro
from construct_dataset import * 
from torchsummary import summary
from evaluation import *
from mutils import Bar, Logger, AverageMeter, mkdir_p, savefig
import time
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
class TrainVQGAN:
    def __init__(self, args, user_id, total_far, total_frr, total_err, total_f1, total_auc):
        self.vqgan = VQGAN(args).to(device=args.device)
        #self.vqgan = torch.nn.parallel.DistributedDataParallel(self.vqgan, device_ids=[0])
        '''
        self.vqgan = DDP(
                self.vqgan,
                device_ids=[torch.device(f"cuda")],
                output_device=torch.device(f"cuda"),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        '''
        self.vqgan = torch.nn.DataParallel(self.vqgan, device_ids=[0,1])
        '''
        self.vqgan.load_checkpoint(args.checkpoint_path)
        self.vqgan = self.vqgan.eval()
        '''
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=[0,1])
        #print("the layer is:", summary(self.discriminator, (1,12,50)))
        #exit(-1)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        self.trainwithoutdis(args, user_id, total_far, total_frr, total_err, total_f1, total_auc)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.module.encoder.parameters()) +
            list(self.vqgan.module.decoder.parameters()) +
            list(self.vqgan.module.codebook.parameters()) +
            list(self.vqgan.module.quant_conv.parameters()) +
            list(self.vqgan.module.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        #train_dataset = load_data(args)
        user_num = 1
        x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_another(cls = user_num)
        #x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_random_attack(cls = user_num)
        #print("---------------the length is:", len(x_valid))
        
        train_dataset = CIFAR10DataLoader_pro(x_train, y_train, split="train", batch_size = args.batch_size, user_num=user_num)
        #test_dataloader = CIFAR10DataLoader_pro(x_valid, y_valid, split="test", batch_size = len(x_valid), user_num=user_num)
        test_dataloader = CIFAR10DataLoader_pro(x_valid, y_valid, split="test", batch_size = 1024, user_num=user_num)
        '''
        train_dataset = CIFAR10DataLoader(
        data_dir='./dataset',
        split='train',
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0)
        '''
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs,_) in zip(pbar, train_dataset):
                    imgs = imgs.type(torch.FloatTensor)
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    #print("#######################", decoded_images.size(), imgs.size())

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    #perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    #perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                #torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))
                
            self.test(args, test_dataloader, self.vqgan)
        torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{user_num}.pt"))    
    

    def trainwithoutdis(self, args, user_id, total_far, total_frr, total_eer, total_f1, total_auc):
        #train_dataset = load_data(args)
        #user_num = 1
        x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_another(cls = user_id)
        self.subjects = subjects
        #x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects = get_mnist_con_random_attack(cls = user_num)
        #print("---------------the length is:", len(x_valid))
        
        train_dataset = CIFAR10DataLoader_pro(x_train, y_train, split="train", batch_size = args.batch_size, user_num=user_id)
        #test_dataloader = CIFAR10DataLoader_pro(x_valid, y_valid, split="test", batch_size = len(x_valid), user_num=user_num)
        test_dataloader = CIFAR10DataLoader_pro(x_valid, y_valid, split="test", batch_size = 128, user_num=user_id)
        '''
        train_dataset = CIFAR10DataLoader(
        data_dir='./dataset',
        split='train',
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0)
        '''
        best_far=10000
        best_frr = 0
        best_eer = 10000
        best_f1 = 0
        best_auc = 0
        steps_per_epoch = len(train_dataset)
        for epoch in tqdm(range(args.epochs), desc="total epoch"):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs,_) in zip(pbar, train_dataset):
                    imgs = imgs.type(torch.FloatTensor)
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    #print("#######################", decoded_images.size(), imgs.size())
                    '''
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)
                    '''

                    #perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    #perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    #g_loss = -torch.mean(disc_fake)

                    #λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss# + disc_factor * λ * g_loss

                    '''
                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)
                    '''
                    self.opt_vq.zero_grad()
                    vq_loss.sum().backward(retain_graph=True)
                    #vq_loss.backward()
                    '''
                    self.opt_disc.zero_grad()
                    gan_loss.backward()
                    '''
                    self.opt_vq.step()
                    #self.opt_disc.step()
                    '''
                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)
                    '''
                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.mean().cpu().detach().numpy().item(), 5),
                        GAN_Loss=0
                    )
                    pbar.update(0)
                #torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))
                
            auroc, far, frr, eer, f1 = self.test(args, test_dataloader, self.vqgan)
            if best_eer > eer:
                best_far = far   
                best_frr = frr
                best_eer = eer
                best_f1 = f1
                best_auc = auroc
        
        total_far.append(best_far)
        total_frr.append(best_frr)
        total_eer.append(best_eer)
        total_f1.append(best_f1)
        total_auc.append(best_auc)
        torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch_{user_id}.pt"))    
            

    def test(self, args, testloader, vqgan):
        global best_acc
    
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        far = AverageMeter()
        frr = AverageMeter()
        eer = AverageMeter()
        f1_score = AverageMeter()
    
        # switch to evaluate mode
        #vqgan = vqgan.cpu()
        vqgan.eval()
        end = time.time()
        bar = Bar('Processing', max=len(testloader))
       
        threshold = 0.0032
        
        
        y_true = []
        y_pre = []
        scores_total = []
    
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.type(torch.FloatTensor)
            #if use_cuda:
            inputs, targets = inputs.to(device=args.device), targets.to(device=args.device)
            # with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            decoded_images, _, q_loss = self.vqgan(inputs)
            #scores = torch.abs(inputs - decoded_images).mean()
            scores = torch.mean(torch.pow((inputs - decoded_images), 2),dim=[1,2,3])
            #print("#############################:", scores.size(), targets.size())
            #scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2,3])+torch.mean(torch.pow((l1 - l1_h), 2),dim=[1,2,3])#+torch.mean(torch.pow((l2 - l2_h), 2),dim=[1,2,3])+torch.mean(torch.pow((l3 - l3_h), 2),dim=[1,2,3])
            '''
            print("the target is:",targets[0:20])
            print("the scores is:",scores[0:20])
            '''
            #scores1 = scores.cpu().detach().numpy()
            #scores2 = -scores.cpu().detach().numpy()
            #scores1[scores1>=threshold] = 0
            #scores1[scores1<threshold] = 1
            '''
            for i in range(len(scores1)):
                if scores1[i] >= threshold:
                    scores1[i] = 0
                else:
                    scores1[i] = 1
            targets1 = targets.cpu().detach().numpy()
            if batch_idx == 0:
                y_true = list(targets1.astype('int64'))
                y_pre = list(scores1.astype('int64'))
                scores_total = list(scores2)
            else:
                y_true = y_true + list(targets1.astype('int64'))
                y_pre = y_pre + list(scores1.astype('int64'))
                scores_total = scores_total + list(scores2)
            '''
            '''
            print("\nthe scores is:", scores.cpu().detach().numpy())
            print("\nthe scores1 is:", scores1)
            print("\nthe y_true is:", targets.cpu().detach().numpy())
            '''
            
            prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())
            
            #fpr,tpr,thresholds = roc_curve(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy(),pos_label=1)
            far_, frr_, eer_, f1_ = utils_eer(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())
            #print("------------------------------frr_:", frr_)
            '''
            print("\nthe auroc is:", prec1)
            print("\nthe fpr is:", fpr)
            print("\nthe tpr is:", tpr)
            print("\nthe thresholds is:", thresholds)
            '''
            top1.update(prec1, inputs.size(0))
            far.update(far_, 1)
            frr.update(frr_, 1)
            eer.update(eer_, 1)
            f1_score.update(f1_, 1)
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # plot progress
            
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | far: {far: .4f} | frr: {frr: .4f} | eer: {eer: .4f} | f1_score: {f1_score: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=top1.avg,
                    far=far.avg,
                    frr=frr.avg,
                    eer=eer.avg,
                    f1_score=f1_score.avg,
                    )
        bar.next()
            
        '''
        fpr,tpr,thresholds = roc_curve(y_true, scores_total,pos_label=1)
        print("\nthe fpr is:", fpr)
        print("\nthe tpr is:", tpr)
        print("\nthe thresholds is:", thresholds)
        '''
        '''
        cfm = confusion_matrix(y_true, y_pre, labels = [0, 1])
        f1_scores = f1_score(y_true, y_pre)
        '''
        #err = utils_eer(y_one_test, yhat1, return_threshold=False)
        #print("\n---------------------------result:")
        #print("the owner is:", owner)
        #print(f'SVM accuracy is: {acc}')
        #print("the err is:", err)
        '''
        print(cfm)
        #np.sum(y_one_test == -1)
        far = cfm[0,1]/ np.sum(np.array(y_true) == 0)
        frr = cfm[1,0]/ np.sum(np.array(y_true) == 1)
        print('f1_score: ', f1_scores, 'FAR: ', far, ' FFR: ', frr)
        '''
        bar.finish()
        
        return top1.avg, far.avg, frr.avg, eer.avg, f1_score.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    #args.dataset_path = r"C:\Users\dome\datasets\flowers"
    #torch.distributed.init_process_group(backend="nccl")
    total_far = []
    total_frr = []
    total_err = []
    total_f1 = []
    total_auc = []
    index = []
    data = pd.DataFrame(columns=['user_id','far', 'frr', 'err', 'f1_score', 'auc'])
    for i in tqdm(range(90), desc="total user"):
        user_id = i
        train_vqgan = TrainVQGAN(args, user_id, total_far, total_frr, total_err, total_f1, total_auc)
        index.append(train_vqgan.subjects[user_id])


    data['user_id'] = index
    data['far'] = total_far
    data['frr'] = total_frr
    data['err'] = total_err
    data['f1_score'] = total_f1
    data['auc'] = total_auc
    data.to_csv('./experiment_results/hmog-nobustrobust-16.csv', sep=',', header=True, index=False,mode='a')



